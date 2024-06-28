import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.metrics.pairwise import cosine_similarity

from .helpers import TextProcessor, Encoder, SummarizationDataset

SENT_SEP = "\n"



class SummarizationPipeline:

	def __init__(
			self, summarizer, encoder: Encoder, max_tokens: int,
			postprocessor:TextProcessor|None=None,
			device: str|torch.device|None=None
		) -> None:
		self.summarizer = summarizer.to(device)
		self.encoder = encoder
		self.max_tokens = max_tokens
		self.postprocessor = postprocessor
		self.device = device

	def __call__(
			self, texts: str|list[str], batch_size: int|None=None
		) -> list[str]:
		if isinstance(texts, str):
			texts = [texts]
		if batch_size is None:
			batch_size = len(texts)
		encoder = self.encoder
		dataset = SummarizationDataset(texts, encoder, batch_size)
		summaries = []
		for encodings in dataset:
			encodings = encodings.to(self.device)
			outputs = self.summarizer.generate(
				**encodings, max_length=self.max_tokens
			)
			summaries.extend([
				encoder.tokenizer.decode(out, skip_special_tokens=True)
				for out in outputs
			])
		if self.postprocessor is not None:
			summaries = self.postprocessor(summaries)
		return summaries
	


class TruncateMiddle(Encoder):

	def __init__(
			self, tokenizer, context_size:int, head_size: float=.5,
			preprocessor: TextProcessor|None=None
		) -> None:
		super().__init__(
			tokenizer, preprocessor,
			tokenizer.bos_token_id, tokenizer.eos_token_id
		)
		self.context_size = context_size
		self.head_size = head_size

	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		context_size = self.context_size - 2
		tokenizer = self.tokenizer
		# Constant head size
		head_size = int(context_size * self.head_size)
		truncated_ids = []

		for text in texts:
			# Encode the text
			encodings = tokenizer.encode(
				text, add_special_tokens=False
			)

			# If the encodings dont fit in the model
			if len(encodings) > context_size:
				# Calculate beginning index of tail
				tail_idx = len(encodings) - context_size + head_size

				# Truncate the middle and concatenate head and tail
				encodings = np.concatenate([
					encodings[:head_size],
					encodings[tail_idx:]
				]).tolist()

			# Add BOS and EOS tokens
			encodings = self.add_special_tokens(encodings)
			truncated_ids.append(encodings)
		
		# Pad sentences and create attention mask
		padded_ids = tokenizer.pad({
			"input_ids": truncated_ids
			}, return_tensors="pt")

		return padded_ids



class UniformSampler(Encoder):

	def __init__(
			self, tokenizer, context_size: int,
			sent_tokenizer, preprocessor: TextProcessor|None=None,
			seed: int|None=None
		) -> None:
		super().__init__(
			tokenizer, preprocessor,
			tokenizer.bos_token_id, tokenizer.eos_token_id
		)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.seed = seed
		np.random.seed(seed)
		self.sent_sep_id = tokenizer.encode(SENT_SEP)[1]

	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		tokenizer = self.tokenizer
		context_size = self.context_size - 2

		processed_texts = []
		for text in texts:
			# Check if encodings fit in the model
			encodings = tokenizer.encode(
				text, add_special_tokens=False
			)
			if len(encodings) <= context_size:
				processed_texts.append(encodings)
				continue

			# Extract and tokenize sentences
			sentences = self.sent_tokenizer(text)
			sentences = tokenizer(
				sentences, add_special_tokens=False
			)["input_ids"]
			sentences = np.array(sentences, dtype=object)

			# Sum of length of sentences
			total_length = sum([
				len(sent) for sent in sentences
			])

			# Approximate probability of picking a sentence
			p = context_size / total_length

			# Sample until sentences fit in model
			while True:
				sent_mask = np.random.rand(len(sentences)) <= p
				sampled = sentences[sent_mask]

				# Flatten sentences
				sampled = [
					elm for lis in sampled
					for elm in lis + [self.sent_sep_id]
				]

				if len(sampled) <= context_size:
					break

			# Add BOS and EOS tokens
			sampled = self.add_special_tokens(sampled)

			# Add sampled sentences to processed texts
			processed_texts.append(sampled)

		# Pad sentences and create attention mask
		padded_ids = tokenizer.pad({
			"input_ids": processed_texts
		}, return_tensors="pt")

		return padded_ids
	


class SentenceSampler(Encoder):

	def __init__(
			self, tokenizer, context_size: int, sent_tokenizer,
			sent_encoder, preprocessor: TextProcessor|None=None,
			threshold: float=.7, device: str|torch.device|None=None,
			seed: int|None=None
		) -> None:
		super().__init__(
			tokenizer, preprocessor,
			tokenizer.bos_token_id, tokenizer.eos_token_id
		)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.sent_encoder = sent_encoder.to(device)
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.device = device
		self.seed = seed
		np.random.seed(seed)
		self.sent_sep_id = tokenizer.encode(SENT_SEP)[1]

	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		sent_tokenizer = self.sent_tokenizer
		tokenizer = self.tokenizer
		context_size = self.context_size - 2

		processed_texts = []
		for text in texts:
			# Check if encodings fit in the model
			encodings = tokenizer.encode(
				text, add_special_tokens=False
			)
			if len(encodings) <= context_size:
				processed_texts.append(encodings)
				continue

			# Extract and tokenize sentences
			sentences = sent_tokenizer(text)
			sentences = tokenizer(
				sentences, add_special_tokens=False
			)["input_ids"]

			# Sum of length of sentences
			total_length = np.sum([
				len(sent) for sent in sentences
			])

			# Approximate probability of picking a sentence
			p = context_size / total_length

			# Sample until sentences fit in model
			while True:
				sampled = []
				sampled_embedding = np.zeros((1, self.sent_embedding_dim))
				num_sampled = 0
				for sent_encoding in sentences:
					if np.random.rand() > p:
						continue
					sent = tokenizer.decode(sent_encoding)
					sent_embedding = self.sent_encoder.encode([sent])
					similarity = cosine_similarity(
						sampled_embedding, sent_embedding
					)
					if self.threshold < similarity:
						continue
					sampled.extend(sent_encoding)
					sampled.append(self.sent_sep_id)
					sampled_embedding = (
						(num_sampled * sampled_embedding + sent_embedding) /
						(num_sampled := num_sampled + 1)
					)
				if len(sampled) <= context_size:
					break
			
			# Add BOS and EOS tokens
			sampled = self.add_special_tokens(sampled)

			# Add sampled sentences to processed texts
			processed_texts.append(sampled)

		# Pad sentences and create attention mask
		padded_ids = tokenizer.pad({
			"input_ids": processed_texts
		}, return_tensors="pt")

		return padded_ids
	


class RemoveRedundancy(Encoder):

	def __init__(
			self, tokenizer, context_size: int, sent_tokenizer,
			sent_encoder, preprocessor: TextProcessor|None=None,
			threshold: float=.7, device: str|torch.device|None=None,
			seed: int|None=None
		) -> None:
		super().__init__(
			tokenizer, preprocessor,
			tokenizer.bos_token_id, tokenizer.eos_token_id
		)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.sent_encoder = sent_encoder.to(device)
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.device = device
		self.seed = seed
		np.random.seed(seed)
		self.sent_sep_id = tokenizer.encode(SENT_SEP)[1]

	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		tokenizer = self.tokenizer
		context_size = self.context_size - 2

		processed_texts = []
		for text in texts:
			# Check if encodings fit in the model
			encodings = tokenizer.encode(
				text, add_special_tokens=False
			)
			if len(encodings) <= context_size:
				processed_texts.append(encodings)
				continue

			# Extract sentences
			sentences = self.sent_tokenizer(text)

			# Remove redundant sentences
			sentences = self.remove_redundancy(sentences)

			# Tokenize sentences
			sentences = tokenizer(
				sentences, add_special_tokens=False
			)["input_ids"]
			sentences = np.array(sentences, dtype=list)

			# Sum of length of sentences
			total_length = sum([
				len(sent) for sent in sentences
			])

			# Approximate probability of picking a sentence
			p = context_size / total_length

			# Sample until sentences fit in model
			while True:

				sent_mask = np.random.rand(len(sentences)) <= p
				sampled = sentences[sent_mask]

				# Flatten sentences
				sampled = [
					elm for lis in sampled
					for elm in lis + [self.sent_sep_id]
				]

				if len(sampled) <= context_size:
					break

			# Add BOS and EOS tokens
			sampled = self.add_special_tokens(sampled)

			# Add sampled sentences to processed texts
			processed_texts.append(sampled)

		# Pad sentences and create attention mask
		padded_ids = tokenizer.pad({
			"input_ids": processed_texts
		}, return_tensors="pt")

		return padded_ids
	
	def remove_redundancy(self, sents: list[str]) -> list[str]:
		selected_sents = []

		# Average embedding of selected sentences
		selected_embedding = np.zeros((1, self.sent_embedding_dim))

		num_sents = 0
		for sent in sents:
			sent_embedding = self.sent_encoder.encode([sent])

			# Calculate similarity between current sentence and chosen sentences
			similarity = cosine_similarity(
				selected_embedding, sent_embedding
			)

			# Discard current sentence and contnue if it is similar
			if self.threshold < similarity:
				continue

			# Otherwise select it
			selected_sents.append(sent)

			# Update selected sentences embedding
			selected_embedding = (
				(num_sents * selected_embedding + sent_embedding) /
				(num_sents := num_sents + 1)
			)
		return selected_sents

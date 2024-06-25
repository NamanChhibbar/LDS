import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.metrics.pairwise import cosine_similarity

from .helpers import TextProcessor, Encoder, SummarizationDataset



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
			summaries.extend(
				[encoder.tokenizer.decode(out) for out in outputs]
			)
		if self.postprocessor is not None:
			summaries = self.postprocessor(summaries)
		return summaries
	


class TruncateMiddle(Encoder):

	def __init__(
			self, tokenizer, context_size:int, head_size: float=.5,
			preprocessor: TextProcessor|None=None
		) -> None:
		super().__init__(tokenizer, preprocessor)
		self.context_size = context_size
		self.head_size = head_size

	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		# Constant head size
		size = self.context_size
		head_size = int(size * self.head_size)
		truncated_ids = []

		for text in texts:
			# Encode the text
			text_ids = self.tokenizer.encode(text)

			# Check if ids fit in model
			if len(text_ids) <= size:
				truncated_ids.append(text_ids)
				continue

			# Calculate beginning index of tail
			tail_idx = len(text_ids) - size + head_size

			# Truncate the middle and concatenate head and tail
			truncated = np.concatenate([
				text_ids[:head_size],
				text_ids[tail_idx:]
			])
			truncated_ids.append(truncated.astype(int))
		
		# Pad sentences and create attention mask
		padded_ids = self.tokenizer.pad({
			"input_ids": truncated_ids
			}, return_tensors="pt")

		return padded_ids



class UniformSampler(Encoder):

	def __init__(
			self, tokenizer, context_size: int,
			sent_tokenizer, bos_id: int, eos_id: int,
			preprocessor: TextProcessor|None=None, seed: int|None=None
		) -> None:
		super().__init__(tokenizer, preprocessor)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.bos_id = bos_id
		self.eos_id = eos_id
		self.seed = seed
		np.random.seed(seed)

	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		processed_texts = []

		for text in texts:
			# Extract and tokenize sentences
			sentences = self.sent_tokenizer(text)
			sentences = self.tokenizer(
				sentences, add_special_tokens=False
			)["input_ids"]
			sentences = np.array(sentences, dtype=object)

			# Sum of length of sentences
			total_length = sum([
				len(sent) for sent in sentences
			])

			# Approximate probability of picking a sentence
			p = self.context_size / total_length

			# Sample until sentences fit in model
			while True:
				sent_mask = np.random.rand(len(sentences)) <= p
				sampled = sentences[sent_mask]

				# Flatten sentences
				sampled = [elm for lis in sampled for elm in lis]

				if len(sampled) <= self.context_size:
					break

			# Add BOS and EOS tokens
			sampled = [self.bos_id] + sampled + [self.eos_id]

			# Add sampled sentences to processed texts
			processed_texts.append(sampled)

		# Pad sentences and create attention mask
		padded_ids = self.tokenizer.pad({
			"input_ids": processed_texts
		}, return_tensors="pt")

		return padded_ids
	


class SentenceSampler(Encoder):

	def __init__(
			self, tokenizer, context_size: int, sent_tokenizer,
			sent_encoder, bos_id: int, eos_id: int, threshold: float=.7,
			preprocessor: TextProcessor|None=None,
			device: str|torch.device|None=None, seed: int|None=None
		) -> None:
		super().__init__(tokenizer, preprocessor)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.sent_encoder = sent_encoder.to(device)
		self.bos_id = bos_id
		self.eos_id = eos_id
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.device = device
		self.seed = seed
		np.random.seed(seed)

	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		sent_tokenizer = self.sent_tokenizer
		tokenizer = self.tokenizer
		context_size = self.context_size

		processed_texts = []
		for text in texts:
			# Extract and tokenize sentences
			sents = sent_tokenizer(text)
			sents = tokenizer(
				sents, add_special_tokens=False
			)["input_ids"]

			# Sum of length of sentences
			total_length = np.sum([
				len(sent) for sent in sents
			])

			# Approximate probability of picking a sentence
			p = context_size / total_length

			# Sample until sentences fit in model
			while True:
				sampled = []
				sampled_embedding = np.zeros((1, self.sent_embedding_dim))
				num_sampled = 0
				for sent in sents:
					if np.random.rand() > p:
						continue
					sent_embedding = self.sent_encoder.encode([sent])
					similarity = cosine_similarity(
						sampled_embedding, sent_embedding
					)
					if self.threshold < similarity:
						continue
					sampled.extend(sent)
					sampled_embedding = (
						(num_sampled * sampled_embedding + sent_embedding) /
						(num_sampled := num_sampled + 1)
					)
				if len(sampled) <= context_size:
					break
			
			# Add BOS and EOS tokens
			sampled = [self.bos_id] + sampled + [self.eos_id]

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
			sent_encoder, bos_id, eos_id, threshold: float=.7,
			preprocessor: TextProcessor|None=None,
			device: str|torch.device|None=None, seed: int|None=None
		) -> None:
		super().__init__(tokenizer, preprocessor)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.sent_encoder = sent_encoder.to(device)
		self.bos_id = bos_id
		self.eos_id = eos_id
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.device = device
		self.seed = seed
		np.random.seed(seed)

	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		processed_texts = []

		for text in texts:
			# Extract sentences
			sents = self.sent_tokenizer(text)

			# Remove redundant sentences
			sents = self.remove_redundancy(sents)

			# Tokenize sentences
			sents = self.tokenizer(
				sents, add_special_tokens=False
			)["input_ids"]
			sents = np.array(sents, dtype=list)

			# Sum of length of sentences
			total_length = sum([
				len(sent) for sent in sents
			])

			# Approximate probability of picking a sentence
			p = self.context_size / total_length

			# Sample until sentences fit in model
			while True:

				sent_mask = np.random.rand(len(sents)) <= p
				sampled = sents[sent_mask]

				# Flatten sentences
				sampled = [elm for lis in sampled for elm in lis]

				if len(sampled) <= self.context_size:
					break

			# Add BOS and EOS tokens
			sampled = [self.bos_id] + sampled + [self.eos_id]

			# Add sampled sentences to processed texts
			processed_texts.append(sampled)

		# Pad sentences and create attention mask
		padded_ids = self.tokenizer.pad({
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

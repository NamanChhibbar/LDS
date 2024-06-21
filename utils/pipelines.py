from abc import ABC, abstractmethod
import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.metrics.pairwise import cosine_similarity


class Encoder(ABC):

	def __init__(self, tokenizer, preprocessor=None) -> None:
		super().__init__()
		self.tokenizer = tokenizer
		self.preprocessor = preprocessor

	def __call__(self, texts: str|list[str]) -> BatchEncoding:
		if isinstance(texts, str):
			texts = [texts]
		if self.preprocessor:
			texts = self.preprocessor(texts)
		encodings = self.generate_encodings(texts)
		return encodings
	
	@abstractmethod
	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		...


class SummarizationPipeline:

	def __init__(
			self, summarizer, encoder: Encoder, max_tokens: int, postprocessor=None,
			device: str|torch.device|None=None
		) -> None:
		self.summarizer = summarizer.to(device)
		self.encoder = encoder
		self.max_tokens = max_tokens
		self.postprocessor = postprocessor
		self.device = device

	def __call__(self, texts: str|list[str]) -> list[str]:
		encoder = self.encoder
		encodings = encoder(texts).to(self.device)
		outputs = self.summarizer.generate(**encodings, max_length=self.max_tokens)
		summaries = [encoder.tokenizer.decode(out) for out in outputs]
		if self.postprocessor:
			summaries = self.postprocessor(summaries)
		return summaries
	

class TruncateMiddle(Encoder):

	def __init__(
			self, tokenizer, context_size:int, head_size: float=.5, preprocessor=None
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
			self, tokenizer, context_size: int, sent_tokenizer, preprocessor=None,
			seed: int|None=None
		) -> None:
		super().__init__(tokenizer, preprocessor)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.seed = seed
		np.random.seed(seed)

	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		processed_texts = []

		for text in texts:
			# Extract and tokenize sentences
			sents = self.sent_tokenizer(text)
			sents = self.tokenizer(sents)["input_ids"]
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

			# Add sampled sentences to processed texts
			processed_texts.append(sampled)

		# Pad sentences and create attention mask
		padded_ids = self.tokenizer.pad({
			"input_ids": processed_texts
		}, return_tensors="pt")

		return padded_ids
	

class SentenceSampler(Encoder):

	def __init__(
			self, tokenizer, context_size: int, sent_tokenizer, sent_encoder,
			threshold: float=.7, preprocessor=None, device: str|torch.device|None=None,
			seed: int|None=None
		) -> None:
		super().__init__(tokenizer, preprocessor)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.sent_encoder = sent_encoder.to(device)
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
			sents = tokenizer(sents)["input_ids"]

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
					similarity = cosine_similarity(sampled_embedding, sent_embedding)
					if self.threshold < similarity:
						continue
					sampled.extend(sent)
					sampled_embedding = (
						(num_sampled * sampled_embedding + sent_embedding) /
						(num_sampled := num_sampled + 1)
					)
				if len(sampled) <= context_size:
					break

			# Add sampled sentences to processed texts
			processed_texts.append(sampled)

		# Pad sentences and create attention mask
		padded_ids = tokenizer.pad({
			"input_ids": processed_texts
		}, return_tensors="pt")

		return padded_ids
	

class RemoveRedundancy(Encoder):

	def __init__(
			self, tokenizer, context_size: int, sent_tokenizer, sent_encoder,
			threshold: float=.7, preprocessor=None, device: str|torch.device|None=None,
			seed: int|None=None
		) -> None:
		super().__init__(tokenizer, preprocessor)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.sent_encoder = sent_encoder.to(device)
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
			sents = self.tokenizer(sents)["input_ids"]
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

			# Add sampled sentences to processed texts
			processed_texts.append(sampled)

		# Pad sentences and create attention mask
		padded_ids = self.tokenizer.pad({
			"input_ids": processed_texts
		}, return_tensors="pt")

		return padded_ids
	
	def remove_redundancy(self, sents: list[str]) -> list[str]:
		selected_sents = []
		selected_embedding = np.zeros((1, self.sent_embedding_dim))
		num_sents = 0
		for sent in sents:
			sent_embedding = self.sent_encoder.encode([sent])
			similarity = cosine_similarity(selected_embedding, sent_embedding)
			if self.threshold < similarity:
				continue
			selected_sents.append(sent)
			selected_embedding = (
				(num_sents * selected_embedding + sent_embedding) /
				(num_sents := num_sents + 1)
			)
		return selected_sents

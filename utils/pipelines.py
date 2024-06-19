from abc import ABC, abstractmethod
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def get_device():
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


class SummarizationPipeline(ABC):

	def __init__(
			self, summarizer, tokenizer, max_tokens: int, preprocessor=None,
			postprocessor=None, device: str|torch.device="cpu"
		):
		self.summarizer = summarizer.to(device)
		self.tokenizer = tokenizer
		self.max_tokens = max_tokens
		self.preprocessor = preprocessor
		self.postprocessor = postprocessor
		self.device = device

	def __call__(self, texts: list[str]):
		if isinstance(texts, str):
			texts = [texts]
		if self.preprocessor:
			texts = self.preprocessor(texts)
		inputs = self.generate_ids(texts).to(self.device)
		outputs = self.summarizer.generate(**inputs, max_length=self.max_tokens)
		summaries = [self.tokenizer.decode(out) for out in outputs]
		if self.postprocessor:
			summaries = self.postprocessor(summaries)
		return summaries
	
	@abstractmethod
	def generate_ids(self, texts: list[str]):
		...
	

class TruncateMiddle(SummarizationPipeline):

	def __init__(
			self, summarizer, tokenizer, max_tokens: int, context_size: int,
			preprocessor=None, postprocessor=None, head_size: float=.5,
			device: str|torch.device="cpu"
		):
		super().__init__(
			summarizer, tokenizer, max_tokens, preprocessor,
			postprocessor, device
		)
		self.context_size = context_size
		self.head_size = head_size

	def generate_ids(self, texts: list[str]):
		# Constant head size
		head_size = int((size := self.context_size) * self.head_size)
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


class UniformSampler(SummarizationPipeline):

	def __init__(
			self, summarizer, tokenizer, max_tokens: int, context_size: int,
			sent_tokenizer, preprocessor=None, postprocessor=None,
			device: str|torch.device="cpu", seed: int|None=None
		):
		super().__init__(
			summarizer, tokenizer, max_tokens, preprocessor,
			postprocessor, device
		)
		self.sent_tokenizer = sent_tokenizer
		self.context_size = context_size
		self.seed = seed
		np.random.seed(seed)

	def generate_ids(self, texts: list[str]):
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
	

class SentenceSampler(SummarizationPipeline):

	def __init__(
			self, summarizer, tokenizer, max_tokens: int, context_size: int,
			sent_tokenizer, sent_encoder, preprocessor=None, postprocessor=None,
			threshold: float=.7, device: str|torch.device="cpu", seed: int|None=None
		):
		super().__init__(
			summarizer, tokenizer, max_tokens, preprocessor,
			postprocessor, device
		)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.sent_encoder = sent_encoder
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.seed = seed
		np.random.seed(seed)

	def generate_ids(self, texts: list[str]):
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
	

class RemoveRedundancy(SummarizationPipeline):

	def __init__(
			self, summarizer, tokenizer, max_tokens: int, context_size: int,
			sent_tokenizer, sent_encoder, preprocessor=None, postprocessor=None,
			threshold: float=.7, device: str|torch.device="cpu", seed: int|None=None
		):
		super().__init__(
			summarizer, tokenizer, max_tokens, preprocessor,
			postprocessor, device
		)
		self.context_size = context_size
		self.sent_tokenizer = sent_tokenizer
		self.sent_encoder = sent_encoder
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.seed = seed
		np.random.seed(seed)

	def generate_ids(self, texts: list[str]):
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
	
	def remove_redundancy(self, sents):
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

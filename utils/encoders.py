from abc import ABC, abstractmethod
from warnings import filterwarnings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers.tokenization_utils_base import BatchEncoding

from .helpers import TextProcessor


filterwarnings("ignore")

SENT_DELIMITER = " "



class Encoder(ABC):
	"""
	Base class for encoders.
	"""
	def __init__(
		self, tokenizer, max_tokens: int,
		preprocessor: TextProcessor|None=None,
		add_special_tokens: bool=True, bos_id: int|None=None,
		eos_id: int|None=None
	) -> None:
		"""
		Base class for encoders. DO NOT instantiate directly.

		## Parameters
		`tokenizer`: Hugging Face tokenizer
		`max_tokens`: Max tokens in text encodings
		`preprocessor`: Text preprocessor
		`add_special_tokens`: Add BOS and EOS tokens to text before
		summary generation
		`bos_id`: Beginning Of Sentence (BOS) token id
		`eos_id`: End Of Sentence (EOS) token id
		"""
		super().__init__()
		self.tokenizer = tokenizer
		self.max_tokens = max_tokens
		self.preprocessor = preprocessor
		self.add_special_tokens = add_special_tokens
		self.bos_id = bos_id
		self.eos_id = eos_id

	def __call__(
		self, texts: str|list[str], max_tokens: int|None=None
	) -> BatchEncoding:
		"""
		Encode texts

		## Parameters
		`texts`: Texts (or text) to encode
		`max_tokens`: Max tokens in text encodings; overrides
		the default value of `max_tokens` if specified

		## Returns
		`encodings`: Text encodings of type BatchEncoding
		"""
		if isinstance(texts, str):
			texts = [texts]
		if max_tokens is None:
			max_tokens = self.max_tokens
		if self.preprocessor is not None:
			texts = self.preprocessor(texts)
		encodings = []
		for text in texts:
			encoded_text = self.encode(text, max_tokens)
			if self.add_special_tokens:
				encoded_text = self.add_tokens(encoded_text)
			encodings.append(encoded_text)
		batch_encoding = self.tokenizer.pad({
			"input_ids": encodings
		}, return_tensors="pt")
		return batch_encoding
	
	@abstractmethod
	def encode(
		self, text: str, max_tokens: int
	) -> list[int]:
		"""
		Creates encdoings for a given text which fit in the
		model's context size.

		## Parameters
		`text`: Text to encode
		`max_tokens`: Max tokens in text encodings

		## Returns
		`encodings`: Text encodings
		"""
		...
	
	def add_tokens(
		self, encodings: list[int]
	) -> list[int]:
		bos_id = self.bos_id
		eos_id = self.eos_id
		if bos_id is not None:
			text_ids = [bos_id] + encodings
		if eos_id is not None:
			text_ids = text_ids + [eos_id]
		return text_ids
	


class TruncateMiddle(Encoder):

	def __init__(
		self, tokenizer, max_tokens: int,
		head_size: float=.5,
		preprocessor: TextProcessor|None=None,
		add_special_tokens: bool=True
	) -> None:
		super().__init__(
			tokenizer, max_tokens, preprocessor, add_special_tokens,
			tokenizer.bos_token_id, tokenizer.eos_token_id
		)
		self.head_size = head_size

	def encode(
		self, text: str, max_tokens: int|None=None
	) -> list[int]:
		tokenizer = self.tokenizer
		if max_tokens is None:
			max_tokens = self.max_tokens
		if self.add_special_tokens:
			max_tokens -= 2

		# Encode the text
		encodings = tokenizer.encode(
			text, add_special_tokens=False
		)

		# Check if encodings fit in the model
		if len(encodings) <= max_tokens:
			return encodings

		# Calculate indices of head and tail
		head_idx = int(max_tokens * self.head_size)
		tail_idx = len(encodings) - max_tokens + head_idx

		# Truncate the middle and concatenate head and tail
		encodings = np.concatenate([
			encodings[:head_idx],
			encodings[tail_idx:]
		]).tolist()

		return encodings



class UniformSampler(Encoder):

	def __init__(
		self, tokenizer, max_tokens: int,
		sent_segmenter, preprocessor: TextProcessor|None=None,
		add_special_tokens: bool=True, seed: int|None=None
	) -> None:
		super().__init__(
			tokenizer, max_tokens, preprocessor, add_special_tokens,
			tokenizer.bos_token_id, tokenizer.eos_token_id
		)
		self.sent_segmenter = sent_segmenter
		self.seed = seed
		np.random.seed(seed)
		self.delimiter_id = tokenizer.encode(
			SENT_DELIMITER, add_special_tokens=False
		)[0]

	def encode(
		self, text: str, max_tokens: int|None=None
	) -> list[int]:
		tokenizer = self.tokenizer
		if max_tokens is None:
			max_tokens = self.max_tokens
		if self.add_special_tokens:
			max_tokens -= 2

		# Check if encodings fit in the model
		encodings = tokenizer.encode(
			text, add_special_tokens=False
		)
		if len(encodings) <= max_tokens:
			return encodings

		# Extract and tokenize sentences
		sentences = self.sent_segmenter(text)
		sentences = tokenizer(
			sentences, add_special_tokens=False
		)["input_ids"]
		sentences = np.array(sentences, dtype=object)

		# Sum of length of sentences
		total_length = sum([
			len(sent) for sent in sentences
		])

		# Approximate probability of picking a sentence
		p = max_tokens / total_length

		# Sample until sentences fit in model
		while True:
			sent_mask = np.random.rand(len(sentences)) <= p
			sampled = sentences[sent_mask]

			# Flatten sentences
			sampled = [
				elm for lis in sampled
				for elm in lis + [self.delimiter_id]
			]
			if len(sampled) <= max_tokens:
				break

		# Remove last sentence separator token
		sampled = sampled[:-1]

		return sampled
	


class SentenceSampler(Encoder):

	def __init__(
		self, tokenizer, max_tokens: int, sent_segmenter,
		sent_encoder, preprocessor: TextProcessor|None=None,
		add_special_tokens: bool=True, threshold: float=.7,
		device: str|torch.device|None=None, seed: int|None=None
	) -> None:
		super().__init__(
			tokenizer, max_tokens, preprocessor, add_special_tokens,
			tokenizer.bos_token_id, tokenizer.eos_token_id
		)
		self.sent_segmenter = sent_segmenter
		self.sent_encoder = sent_encoder.to("cpu")
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.device = device
		self.seed = seed
		np.random.seed(seed)
		self.delimiter_id = tokenizer.encode(
			SENT_DELIMITER, add_special_tokens=False
		)[0]

	def encode(
		self, text: str, max_tokens: int|None=None
	) -> list[int]:
		sent_segmenter = self.sent_segmenter
		sent_encoder = self.sent_encoder.to(self.device)
		if max_tokens is None:
			max_tokens = self.max_tokens
		tokenizer = self.tokenizer
		if self.add_special_tokens:
			max_tokens -= 2

		# Check if encodings fit in the model
		encodings = tokenizer.encode(
			text, add_special_tokens=False
		)
		if len(encodings) <= max_tokens:
			return encodings

		# Extract and tokenize sentences
		sentences = sent_segmenter(text)
		sentences = tokenizer(
			sentences, add_special_tokens=False
		)["input_ids"]

		# Sum of length of sentences
		total_length = np.sum([
			len(sent) for sent in sentences
		])

		# Approximate probability of picking a sentence
		p = max_tokens / total_length

		# Sample until sentences fit in model
		while True:
			sampled = []
			sampled_embedding = np.zeros((1, self.sent_embedding_dim))
			num_sampled = 0
			for sent_encoding in sentences:
				if np.random.rand() > p:
					continue
				sent = tokenizer.decode(sent_encoding)
				sent_embedding = sent_encoder.encode([sent])
				similarity = cosine_similarity(
					sampled_embedding, sent_embedding
				)
				if self.threshold < similarity:
					continue
				sampled.extend(sent_encoding)
				sampled.append(self.delimiter_id)
				sampled_embedding = (
					(num_sampled * sampled_embedding + sent_embedding) /
					(num_sampled := num_sampled + 1)
				)
			if len(sampled) <= max_tokens:
				break

		# Remove sentence encoder from device
		sent_encoder.to("cpu")

		# Remove last sentence separator token
		sampled = sampled[:-1]

		return sampled
	


class RemoveRedundancy(Encoder):

	def __init__(
		self, tokenizer, max_tokens: int, sent_segmenter,
		sent_encoder, preprocessor: TextProcessor|None=None,
		add_special_tokens: bool=True, threshold: float=.7,
		device: str|torch.device|None=None, seed: int|None=None
	) -> None:
		super().__init__(
			tokenizer, max_tokens, preprocessor, add_special_tokens,
			tokenizer.bos_token_id, tokenizer.eos_token_id
		)
		self.sent_segmenter = sent_segmenter
		self.sent_encoder = sent_encoder.to("cpu")
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.device = device
		self.seed = seed
		np.random.seed(seed)
		self.delimiter_id = tokenizer.encode(
			SENT_DELIMITER, add_special_tokens=False
		)[0]

	def encode(
		self, text: str, max_tokens: int|None=None
	) -> list[int]:
		tokenizer = self.tokenizer
		if max_tokens is None:
			max_tokens = self.max_tokens
		if self.add_special_tokens:
			max_tokens -= 2

		# Check if encodings fit in the model
		encodings = tokenizer.encode(
			text, add_special_tokens=False
		)
		if len(encodings) <= max_tokens:
			return encodings

		# Extract sentences
		sentences = self.sent_segmenter(text)

		# Remove redundant sentences
		sentences = self.remove_redundancy(sentences)

		# Tokenize sentences
		sentences = tokenizer(
			sentences, add_special_tokens=False
		)["input_ids"]
		sentences = np.array(sentences, dtype=object)

		# Sum of length of sentences
		total_length = sum([
			len(sent) for sent in sentences
		])

		# Approximate probability of picking a sentence
		p = max_tokens / total_length

		# Sample until sentences fit in model
		while True:
			sent_mask = np.random.rand(len(sentences)) <= p
			sampled = sentences[sent_mask]

			# Flatten sentences
			sampled = [
				elm for lis in sampled
				for elm in lis + [self.delimiter_id]
			]
			if len(sampled) <= max_tokens:
				break

		# Remove last sentence separator token
		sampled = sampled[:-1]

		return sampled
	
	def remove_redundancy(self, sents: list[str]) -> list[str]:
		sent_encoder = self.sent_encoder.to(self.device)
		selected_sents = []

		# Average embedding of selected sentences
		selected_embedding = np.zeros((1, self.sent_embedding_dim))

		num_sents = 0
		for sent in sents:
			sent_embedding = sent_encoder.encode([sent])

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

		# Remove sentence encoder from device
		sent_encoder.to("cpu")

		return selected_sents

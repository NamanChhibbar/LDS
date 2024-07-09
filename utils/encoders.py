from abc import ABC, abstractmethod
from warnings import filterwarnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers.tokenization_utils_base import BatchEncoding

from .helpers import TextProcessor


filterwarnings("ignore")

SEG_DELIMITER = " "



class Encoder(ABC):
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
	def __init__(
		self, tokenizer, max_tokens: int,
		preprocessor: TextProcessor|None=None,
		add_special_tokens: bool=True, bos_id: int|None=None,
		eos_id: int|None=None, num_workers: int=0
	) -> None:
		super().__init__()
		self.tokenizer = tokenizer
		self.max_tokens = max_tokens
		self.preprocessor = preprocessor
		self.add_special_tokens = add_special_tokens
		self.bos_id = bos_id
		self.eos_id = eos_id
		self.num_workers = num_workers

	def __call__(
		self, texts: str|list[str], max_tokens: int|None=None
	) -> BatchEncoding:
		"""
		Encodes texts to fit in the model's context size and creates a BatchEncoding.

		## Parameters
		`texts`: Texts (or text) to encode
		`max_tokens`: Max tokens in text encodings; overrides the default
		value of `max_tokens` if specified

		## Returns
		`encodings`: Text encodings of type BatchEncoding
		"""
		preprocessor = self.preprocessor
		num_workers = self.num_workers
		if isinstance(texts, str):
			texts = [texts]
		if max_tokens is None:
			max_tokens = self.max_tokens
		if preprocessor is not None:
			texts = preprocessor(texts)
		if num_workers > 1:
			with ProcessPoolExecutor(num_workers) as executor:
				args = [(text, max_tokens) for text in texts]
				all_encodings = executor.map(self._encode_wrapper, args)
				all_encodings = list(all_encodings)
		else:
			all_encodings = [
				self._encode_wrapper((text, max_tokens))
				for text in texts
			]
		batch_encoding = self.tokenizer.pad({
			"input_ids": all_encodings
		}, return_tensors="pt")
		return batch_encoding

	def _encode_wrapper(self, args):
		text, max_tokens = args
		encodings = self.encode(text, max_tokens)
		if self.add_special_tokens:
			encodings = self.add_tokens(encodings)
		return encodings
	
	@abstractmethod
	def encode(
		self, text: str, max_tokens: int
	) -> list[int]:
		"""
		Creates encodings for a given text which fit in the model's context size.

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
			encodings = [bos_id] + encodings
		if eos_id is not None:
			encodings = encodings + [eos_id]
		return encodings
	


class TruncateMiddle(Encoder):

	def __init__(
		self, tokenizer, max_tokens: int, head_size: float=.5,
		preprocessor: TextProcessor|None=None,
		add_special_tokens: bool=True, num_workers: int=0
	) -> None:
		super().__init__(
			tokenizer, max_tokens, preprocessor, add_special_tokens,
			tokenizer.bos_token_id, tokenizer.eos_token_id, num_workers
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
		add_special_tokens: bool=True, seed: int|None=None,
		num_workers: int=0
	) -> None:
		super().__init__(
			tokenizer, max_tokens, preprocessor, add_special_tokens,
			tokenizer.bos_token_id, tokenizer.eos_token_id, num_workers
		)
		self.sent_segmenter = sent_segmenter
		self.seed = seed
		np.random.seed(seed)

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
		num_tokens = len(encodings)
		if num_tokens <= max_tokens:
			return encodings

		# Extract and tokenize sentences
		sentences = self.sent_segmenter(text)
		sentences = np.array(sentences)
		num_sentences = len(sentences)

		# Approximate probability of picking a sentence
		p = max_tokens / num_tokens

		# Sample until sentences fit in model
		while True:
			sent_mask = np.random.rand(num_sentences) <= p
			sampled = sentences[sent_mask]

			# Flatten sentences
			sampled = SEG_DELIMITER.join(sampled)

			# Tokenize sampled sentences
			sampled = tokenizer.encode(
				sampled, add_special_tokens=False
			)

			if len(sampled) <= max_tokens:
				break

		return sampled
	


class SentenceSampler(Encoder):

	def __init__(
		self, tokenizer, max_tokens: int, sent_segmenter,
		sent_encoder, preprocessor: TextProcessor|None=None,
		add_special_tokens: bool=True, threshold: float=.7,
		seed: int|None=None, num_workers: int=0
	) -> None:
		super().__init__(
			tokenizer, max_tokens, preprocessor, add_special_tokens,
			tokenizer.bos_token_id, tokenizer.eos_token_id, num_workers
		)
		self.sent_segmenter = sent_segmenter
		self.sent_encoder = sent_encoder.to("cpu")
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.seed = seed
		np.random.seed(seed)

	def encode(
		self, text: str, max_tokens: int|None=None
	) -> list[int]:
		sent_segmenter = self.sent_segmenter
		sent_encoder = self.sent_encoder
		if max_tokens is None:
			max_tokens = self.max_tokens
		tokenizer = self.tokenizer
		if self.add_special_tokens:
			max_tokens -= 2

		# Check if encodings fit in the model
		encodings = tokenizer.encode(
			text, add_special_tokens=False
		)
		num_tokens = len(encodings)
		if num_tokens <= max_tokens:
			return encodings

		# Extract and tokenize sentences
		sentences = sent_segmenter(text)

		# Approximate probability of picking a sentence
		p = max_tokens / num_tokens

		# Sample until sentences fit in model
		num_iters = 0
		while True:
			num_iters += 1
			sampled = []
			sampled_embedding = np.zeros((1, self.sent_embedding_dim))
			num_sampled = 0

			for sentence in sentences:
				if np.random.rand() > p:
					continue
				sent_embedding = sent_encoder.encode([sentence])
				similarity = cosine_similarity(
					sampled_embedding, sent_embedding
				)
				if self.threshold < similarity:
					continue
				sampled.append(sentence)
				sampled_embedding = (
					(num_sampled * sampled_embedding + sent_embedding) /
					(num_sampled + 1)
				)
				num_sampled += 1
			
			# Flatten sentences
			sampled = SEG_DELIMITER.join(sampled)

			# Tokenize sampled sentences
			sampled = tokenizer.encode(
				sampled, add_special_tokens=False
			)

			if len(sampled) <= max_tokens:
				break

		return sampled
	


class RemoveRedundancy(Encoder):

	def __init__(
		self, tokenizer, max_tokens: int, sent_segmenter,
		sent_encoder, preprocessor: TextProcessor|None=None,
		add_special_tokens: bool=True, threshold: float=.7,
		seed: int|None=None, num_workers: int=0
	) -> None:
		super().__init__(
			tokenizer, max_tokens, preprocessor, add_special_tokens,
			tokenizer.bos_token_id, tokenizer.eos_token_id, num_workers
		)
		self.sent_segmenter = sent_segmenter
		self.sent_encoder = sent_encoder.to("cpu")
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.seed = seed
		np.random.seed(seed)

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
		num_sentences = len(sentences)

		# Tokenize sentences
		tokenized_sentences = tokenizer(
			sentences, add_special_tokens=False
		)["input_ids"]

		# Sum of number of tokens in sentences
		num_tokens = sum([
			len(sent) for sent in tokenized_sentences
		])

		# Approximate probability of picking a sentence
		p = max_tokens / num_tokens

		# Convert list of sentences to numpy array for sampling
		sentences = np.array(sentences)

		# Sample until sentences fit in model
		while True:
			sent_mask = np.random.rand(num_sentences) <= p
			sampled = sentences[sent_mask]

			# Flatten sentences
			sampled = SEG_DELIMITER.join(sampled)

			# Tokenize sampled sentences
			sampled = tokenizer.encode(
				sampled, add_special_tokens=False
			)

			if len(sampled) <= max_tokens:
				break

		return sampled
	
	def remove_redundancy(self, sents: list[str]) -> list[str]:
		sent_encoder = self.sent_encoder
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

		return selected_sents

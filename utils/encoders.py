from abc import ABC, abstractmethod
from warnings import filterwarnings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
		eos_id: int|None=None
	) -> None:
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
		Encodes texts to fit in the model's context size and creates a BatchEncoding.

		## Parameters
		`texts`: Texts (or text) to encode
		`max_tokens`: Max tokens in text encodings; overrides the default
		value of `max_tokens` if specified

		## Returns
		`BatchEncoding`: Batched text encodings
		"""
		preprocessor = self.preprocessor
		if isinstance(texts, str):
			texts = [texts]
		if max_tokens is None:
			max_tokens = self.max_tokens
		if preprocessor is not None:
			texts = preprocessor(texts)
		encodings = [
			self._encode_wrapper(text, max_tokens) for text in texts
		]
		batch_encoding = self.tokenizer.pad({
			"input_ids": encodings
		}, return_tensors="pt")
		return batch_encoding
	
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
		`list[int]`: Text encodings
		"""
		...

	def _encode_wrapper(self, text, max_tokens):
		if self.add_special_tokens:
			max_tokens -= 2
		encoding = self.encode(text, max_tokens)
		if self.add_special_tokens:
			encoding = self.add_tokens(encoding)
		return encoding
	
	def add_tokens(
		self, encoding: list[int]
	) -> list[int]:
		bos_id = self.bos_id
		eos_id = self.eos_id
		if bos_id is not None:
			encoding = [bos_id] + encoding
		if eos_id is not None:
			encoding = encoding + [eos_id]
		return encoding
	


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

		# Encode the text
		encoding = tokenizer.encode(
			text, add_special_tokens=False
		)
		encoding_size = len(encoding)

		# Check if encodings fit in the model
		if encoding_size <= max_tokens:
			return encoding

		# Calculate indices of head and tail
		head_idx = int(max_tokens * self.head_size)
		tail_idx = encoding_size - max_tokens + head_idx

		# Truncate the middle and concatenate head and tail
		encoding = np.concatenate([
			encoding[:head_idx],
			encoding[tail_idx:]
		]).tolist()

		return encoding



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

		# Check if encodings fit in the model
		encoding = tokenizer.encode(
			text, add_special_tokens=False
		)
		encoding_size = len(encoding)
		if encoding_size <= max_tokens:
			return encoding

		# Extract and tokenize segments
		segments = self.sent_segmenter(text)
		segments = np.array(segments)
		num_segments = len(segments)

		# Approximate probability of picking a segment
		p = max_tokens / encoding_size

		# Sample until segments fit in model
		while True:
			# Create sampling mask
			sent_mask = np.random.rand(num_segments) <= p
			sampled = segments[sent_mask]

			# Join sampled segments with delimiter
			sampled = SEG_DELIMITER.join(sampled)

			# Tokenize sampled segments
			sampled = tokenizer.encode(
				sampled, add_special_tokens=False
			)

			# Check if sampled segments fit in model
			if len(sampled) <= max_tokens:
				break

		return sampled
	


class SegmentSampler(Encoder):

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
		self.sent_encoder = sent_encoder
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

		# Check if encodings fit in the model
		encodings = tokenizer.encode(
			text, add_special_tokens=False
		)
		num_tokens = len(encodings)
		if num_tokens <= max_tokens:
			return encodings

		# Extract and tokenize segments
		segments = sent_segmenter(text)

		# Approximate probability of picking a segment
		p = max_tokens / num_tokens

		# Sample until segments fit in model
		num_iters = 0
		while True:
			num_iters += 1
			sampled = []
			sampled_embedding = np.zeros((1, self.sent_embedding_dim))
			num_sampled = 0

			for segment in segments:
				if np.random.rand() > p:
					continue
				sent_embedding = sent_encoder.encode([segment])
				similarity = cosine_similarity(
					sampled_embedding, sent_embedding
				)
				if self.threshold < similarity:
					continue
				sampled.append(segment)
				sampled_embedding = (
					(num_sampled * sampled_embedding + sent_embedding) /
					(num_sampled + 1)
				)
				num_sampled += 1
			
			# Flatten segments
			sampled = SEG_DELIMITER.join(sampled)

			# Tokenize sampled segments
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
		self.sent_encoder = sent_encoder
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

		# Check if encodings fit in the model
		encodings = tokenizer.encode(
			text, add_special_tokens=False
		)
		if len(encodings) <= max_tokens:
			return encodings

		# Extract segments
		segments = self.sent_segmenter(text)

		# Remove redundant segments
		segments = self.remove_redundancy(segments)
		num_segments = len(segments)

		# Tokenize segments
		tokenized_segments = tokenizer(
			segments, add_special_tokens=False
		)["input_ids"]

		# Sum of number of tokens in segments
		num_tokens = sum([
			len(sent) for sent in tokenized_segments
		])

		# Approximate probability of picking a segment
		p = max_tokens / num_tokens

		# Convert list of segments to numpy array for sampling
		segments = np.array(segments)

		# Sample until segments fit in model
		while True:
			sent_mask = np.random.rand(num_segments) <= p
			sampled = segments[sent_mask]

			# Flatten segments
			sampled = SEG_DELIMITER.join(sampled)

			# Tokenize sampled segments
			sampled = tokenizer.encode(
				sampled, add_special_tokens=False
			)

			if len(sampled) <= max_tokens:
				break

		return sampled
	
	def remove_redundancy(self, sents: list[str]) -> list[str]:
		sent_encoder = self.sent_encoder
		selected_sents = []

		# Average embedding of selected segments
		selected_embedding = np.zeros((1, self.sent_embedding_dim))

		num_sents = 0
		for sent in sents:
			sent_embedding = sent_encoder.encode([sent])

			# Calculate similarity between current segment and chosen segments
			similarity = cosine_similarity(
				selected_embedding, sent_embedding
			)

			# Discard current segment and contnue if it is similar
			if self.threshold < similarity:
				continue

			# Otherwise select it
			selected_sents.append(sent)

			# Update selected segments embedding
			selected_embedding = (
				(num_sents * selected_embedding + sent_embedding) /
				(num_sents := num_sents + 1)
			)

		return selected_sents

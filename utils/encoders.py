from abc import ABC, abstractmethod
from warnings import filterwarnings
from typing import Callable

import numpy as np
from transformers.tokenization_utils_base import BatchEncoding
from sentence_transformers import SentenceTransformer

from .helpers import count_tokens, get_keywords


filterwarnings("ignore")



class Encoder(ABC):
	"""
	Base class for encoders.

	## Parameters
	`tokenizer`: Hugging Face tokenizer
	`min_tokens`: Min tokens in text encodings
	`max_tokens`: Max tokens in text encodings
	`preprocessor`: Text preprocessor
	`add_special_tokens`: Add BOS and EOS tokens to text before
	summary generation
	`bos_id`: Beginning Of Sentence (BOS) token id
	`eos_id`: End Of Sentence (EOS) token id
	"""
	def __init__(
		self,
		tokenizer,
		min_tokens: int,
		max_tokens: int,
		preprocessor: Callable[[list[str]], list[str]] | None = None,
		add_special_tokens: bool = True,
		bos_id: int | None = None,
		eos_id: int | None = None
	) -> None:
		self.tokenizer = tokenizer
		self.min_tokens = min_tokens
		self.max_tokens = max_tokens
		self.preprocessor = preprocessor
		self.add_special_tokens = add_special_tokens
		self.bos_id = bos_id
		self.eos_id = eos_id
		self.num_special_tokens = \
			int(bos_id is not None) + int(eos_id is not None)

	def __call__(
		self,
		texts: str | list[str],
		min_tokens: int | None = None,
		max_tokens: int | None = None,
		return_batch: bool = True
	) -> list[int] | list[list[int]] | BatchEncoding:
		"""
		Encodes texts to fit in the model's context size and creates a BatchEncoding.

		## Parameters
		`texts`: Texts (or text) to encode
		`min_tokens`: Min tokens in text encodings; overrides the default
		value of `min_tokens` if specified
		`max_tokens`: Max tokens in text encodings; overrides the default
		value of `max_tokens` if specified

		## Returns
		`BatchEncoding`: Batched text encodings
		"""
		preprocessor = self.preprocessor
		if isinstance(texts, str):
			texts = [texts]
		if min_tokens is None:
			min_tokens = self.min_tokens
		if max_tokens is None:
			max_tokens = self.max_tokens
		if preprocessor is not None:
			texts = preprocessor(texts)
		encodings = [
			self._encode_wrapper(text, min_tokens, max_tokens)
			for text in texts
		]
		if return_batch:
			encodings = self.tokenizer.pad({
				"input_ids": encodings
			}, return_tensors="pt")
		return encodings
	
	@abstractmethod
	def encode(
		self,
		text: str,
		min_tokens: int | None = None,
		max_tokens: int | None = None
	) -> list[int]:
		"""
		Creates encoding for a given text with number of tokens
		in the range [`min_tokens`, `max_tokens`].

		## Parameters
		`text`: Text to encode
		`min_tokens`: Minimum tokens in text encodings
		`max_tokens`: Maximum tokens in text encodings

		## Returns
		`list[int]`: Text encodings
		"""
		pass

	def _encode_wrapper(
		self,
		text: str,
		min_tokens: int,
		max_tokens: int
	) -> list[int]:
		if self.add_special_tokens:
			max_tokens -= self.num_special_tokens
		num_tokens, encoding = count_tokens(text, self.tokenizer)
		if num_tokens > max_tokens:
			encoding = self.encode(text, min_tokens, max_tokens)
		if self.add_special_tokens:
			encoding = self.add_tokens(encoding)
		return encoding
	
	def add_tokens(
		self,
		encoding: list[int]
	) -> list[int]:
		bos_id = self.bos_id
		eos_id = self.eos_id
		if bos_id is not None:
			encoding = [bos_id] + encoding
		if eos_id is not None:
			encoding = encoding + [eos_id]
		return encoding
	


class VanillaEncoder(Encoder):

	def __init__(
		self,
		tokenizer,
		max_tokens: int,
		preprocessor: Callable[[list[str]], list[str]] | None = None,
		add_special_tokens: bool = True,
		bos_id: int | None = None,
		eos_id: int | None = None
	) -> None:
		super().__init__(
			tokenizer, 0, max_tokens, preprocessor,
			add_special_tokens, bos_id, eos_id
		)

	def encode(
		self,
		text: str,
		_ = None,
		max_tokens: int | None = None
	) -> list[int]:
		max_tokens = max_tokens or self.max_tokens
		tokenizer = self.tokenizer

		# Encode the text
		encoding = tokenizer.encode(
			text,
			max_length=max_tokens,
			truncation=True,
			add_special_tokens=False
		)
		return encoding
	


class TruncateMiddle(Encoder):

	def __init__(
		self,
		tokenizer,
		max_tokens: int,
		head_size: float = .5,
		preprocessor: Callable[[list[str]], list[str]] | None = None,
		add_special_tokens: bool = True
	) -> None:
		super().__init__(
			tokenizer, 0, max_tokens, preprocessor,
			add_special_tokens, tokenizer.bos_token_id,
			tokenizer.eos_token_id
		)
		self.head_size = head_size

	def encode(
		self,
		text: str,
		_ = None,
		max_tokens: int | None = None
	) -> list[int]:
		tokenizer = self.tokenizer
		max_tokens = max_tokens or self.max_tokens

		# Encode the text
		num_tokens, encoding = count_tokens(text, tokenizer)

		# Calculate indices of head and tail
		head_idx = int(max_tokens * self.head_size)
		tail_idx = num_tokens - max_tokens + head_idx

		# Truncate the middle and concatenate head and tail
		encoding = np.concatenate([
			encoding[:head_idx],
			encoding[tail_idx:]
		]).tolist()

		return encoding



class UniformSampler(Encoder):

	def __init__(
		self,
		tokenizer,
		min_tokens: int,
		max_tokens: int,
		text_segmenter: Callable[[str], list[str]],
		preprocessor: Callable[[list[str]], list[str]] | None = None,
		add_special_tokens: bool = True,
		segment_delimiter: str = " ",
		seed: int | None = None,
	) -> None:
		super().__init__(
			tokenizer, min_tokens, max_tokens,
			preprocessor, add_special_tokens,
			tokenizer.bos_token_id, tokenizer.eos_token_id
		)
		self.text_segmenter = text_segmenter
		self.segment_delimiter = segment_delimiter
		self.seed = seed
		np.random.seed(seed)

	def encode(
		self,
		text: str,
		min_tokens: int | None = None,
		max_tokens: int | None = None
	) -> list[int]:
		tokenizer = self.tokenizer
		min_tokens = min_tokens or self.min_tokens
		max_tokens = max_tokens or self.max_tokens

		# Check if encodings fit in the model
		encoding_size, _ = count_tokens(text, tokenizer)

		# Extract and tokenize segments
		segments = self.text_segmenter(text)
		segments = np.array(segments)
		num_segments = len(segments)

		# Approximate probability of picking a segment
		p = max_tokens / encoding_size

		# Sample until segments fit in model
		while True:
			# Create sampling mask
			segment_mask = np.random.rand(num_segments) <= p
			sampled = segments[segment_mask]

			# Flatten and tokenize sampled segments
			flattened = self.segment_delimiter.join(sampled)
			flattened = tokenizer.encode(
				flattened, add_special_tokens=False
			)

			# Break if number of tokens is in range
			if min_tokens <= len(sampled) <= max_tokens:
				break

		return sampled
	


class SegmentSampler(Encoder):

	def __init__(
		self,
		tokenizer,
		min_tokens: int,
		max_tokens: int,
		text_segmenter: Callable[[str], list[str]],
		sent_encoder: SentenceTransformer,
		preprocessor: Callable[[list[str]], list[str]] | None = None,
		add_special_tokens: bool = True,
		threshold: float = .7,
		prob_boost: float = .02,
		segment_delimiter: str = " ",
		seed: int | None = None
	) -> None:
		super().__init__(
			tokenizer, min_tokens, max_tokens, preprocessor,
			add_special_tokens, tokenizer.bos_token_id,
			tokenizer.eos_token_id
		)
		self.text_segmenter = text_segmenter
		self.sent_encoder = sent_encoder
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.prob_boost = prob_boost
		self.segment_delimiter = segment_delimiter
		self.seed = seed
		np.random.seed(seed)

	def encode(
		self,
		text: str,
		min_tokens: int | None = None,
		max_tokens: int | None = None
	) -> list[int]:
		tokenizer = self.tokenizer
		text_segmenter = self.text_segmenter
		sent_encoder = self.sent_encoder
		min_tokens = min_tokens or self.min_tokens
		max_tokens = max_tokens or self.max_tokens

		# Extract and tokenize segments
		segments = text_segmenter(text)

		# Approximate probability of picking a segment
		num_tokens, _ = count_tokens(text, tokenizer)
		p = (1 + self.prob_boost) * max_tokens / num_tokens

		# Sample until segments fit in model
		num_iters = 0
		while True:
			num_iters += 1
			flattened = []

			# Initialize sampled embedding
			num_sampled = 0
			sampled_embedding = np.zeros(self.sent_embedding_dim)

			for segment in segments:

				# Randomly sample segments
				if np.random.rand() > p:
					continue

				# Get segment embedding
				segment_embedding = sent_encoder.encode(segment)

				# Calculate similarity between sampled and current segment
				similarity = sampled_embedding @ segment_embedding

				# Continue if current segment is similar
				if self.threshold < similarity:
					continue

				flattened.append(segment)

				# Update sampled embedding
				sampled_embedding = (
					(num_sampled * sampled_embedding + segment_embedding) /
					(num_sampled + 1)
				)
				num_sampled += 1
			
			# Flatten and tokenize sampled segments
			flattened = self.segment_delimiter.join(flattened)
			flattened = tokenizer.encode(
				flattened, add_special_tokens=False
			)

			# Break if number of tokens is in range
			if min_tokens <= len(flattened) <= max_tokens:
				break

		return flattened
	


class RemoveRedundancy(Encoder):

	def __init__(
		self,
		tokenizer,
		min_tokens: int,
		max_tokens: int,
		text_segmenter: Callable[[str], list[str]],
		sent_encoder: SentenceTransformer,
		preprocessor: Callable[[list[str]], list[str]] | None = None,
		add_special_tokens: bool = True,
		threshold: float = .7,
		segment_delimiter: str = " ",
		seed: int | None = None
	) -> None:
		super().__init__(
			tokenizer, min_tokens, max_tokens, preprocessor,
			add_special_tokens, tokenizer.bos_token_id,
			tokenizer.eos_token_id
		)
		self.text_segmenter = text_segmenter
		self.sent_encoder = sent_encoder
		self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
		self.threshold = threshold
		self.segment_delimiter = segment_delimiter
		self.seed = seed
		np.random.seed(seed)

	def encode(
		self,
		text: str,
		min_tokens: int | None = None,
		max_tokens: int | None = None
	) -> list[int]:
		tokenizer = self.tokenizer
		min_tokens = min_tokens or self.min_tokens
		max_tokens = max_tokens or self.max_tokens

		# Extract segments
		segments = self.text_segmenter(text)

		# Remove redundant segments
		segments = self.remove_redundancy(segments)
		num_segments = len(segments)

		# Tokenize segments
		tokenized_segments = tokenizer(
			segments, add_special_tokens=False
		)["input_ids"]

		# Sum of number of tokens in segments
		num_tokens = sum([
			len(segment)
			for segment in tokenized_segments
		])

		# Approximate probability of picking a segment
		p = max_tokens / num_tokens

		# Convert list of segments to numpy array for sampling
		segments = np.array(segments)

		# Sample until segments fit in model
		while True:
			segment_mask = np.random.rand(num_segments) <= p
			sampled = segments[segment_mask]

			# Flatten segments
			flattened = self.segment_delimiter.join(sampled)
			flattened = tokenizer.encode(
				flattened, add_special_tokens=False
			)

			# Break if number of tokens is in range
			if min_tokens <= len(flattened) <= max_tokens:
				break

		return flattened
	
	def remove_redundancy(
		self,
		segments: list[str]
	) -> list[str]:
		sent_encoder = self.sent_encoder
		selected_segments = []

		# Average embedding of selected segments
		selected_embedding = np.zeros(self.sent_embedding_dim)

		num_segments = 0
		for segment in segments:

			# Get segment embedding
			segment_embedding = sent_encoder.encode(segment)

			# Calculate similarity between current segment and chosen segments
			similarity = selected_embedding @ segment_embedding

			# Discard current segment and contnue if it is similar
			if self.threshold < similarity:
				continue

			# Otherwise select it
			selected_segments.append(segment)

			# Update selected segments embedding
			selected_embedding = (
				(num_segments * selected_embedding + segment_embedding) /
				(num_segments := num_segments + 1)
			)
		return selected_segments
	


class KeywordScorer(Encoder):

	def __init__(
		self,
		tokenizer,
		max_tokens: int,
		text_segmenter: Callable[[str], list[str]],
		sent_encoder: SentenceTransformer,
		preprocessor: Callable[[list[str]], list[str]] | None = None,
		keywords_preprocessor: Callable[[list[str]], list[str]] | None = None,
		stop_words: list[str] | None = None,
		add_special_tokens: bool = True,
		segment_delimiter: str = " "
	) -> None:
		super().__init__(
			tokenizer, 0, max_tokens, preprocessor,
			add_special_tokens, tokenizer.bos_token_id,
			tokenizer.eos_token_id
		)
		self.text_segmenter = text_segmenter
		self.sent_encoder = sent_encoder
		self.keywords_preprocessor = keywords_preprocessor
		self.stop_words = stop_words
		self.segment_delimiter = segment_delimiter

	def encode(
		self,
		text: str,
		_ = None,
		max_tokens: int | None = None
	) -> list[str]:
		tokenizer = self.tokenizer
		max_tokens = max_tokens or self.max_tokens
		sent_encoder = self.sent_encoder

		# Extract keywords from the text
		keywords = get_keywords(
			text,
			stop_words = self.stop_words,
			preprocessor = self.keywords_preprocessor
		)
		# Create keywords embedding
		keywords_emb = sent_encoder.encode(" ".join(keywords))

		# Extract segments from the text
		segments = self.text_segmenter(text)

		# Get segment embeddings
		segment_embeddings = sent_encoder.encode(segments)
		
		# Calculate similarity of keywords with each segment
		segment_similarities = []
		for embedding in segment_embeddings:
			similarity = keywords_emb @ embedding
			segment_similarities.append(similarity)
		
		# Argument sort the similarities
		best_segments = np.argsort(segment_similarities)[::-1]

		# Get number of tokens in segments
		segment_lengths = [
			count_tokens(seg, tokenizer)[0]
			for seg in segments
		]

		# Select maximum segment indices with highest scores
		selected_indices = []
		tokens_used = 0
		for i in best_segments:
			if tokens_used + segment_lengths[i] > max_tokens:
				continue
			selected_indices.append(i)
			tokens_used += segment_lengths[i]
		selected_indices.sort()
		
		# Flatten and tokenize selected segments
		flattened = self.segment_delimiter.join([
			segments[i]
			for i in selected_indices
		])
		flattened = tokenizer.encode(
			flattened,
			add_special_tokens=False
		)
		return flattened

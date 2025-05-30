'''
Contains callable encoder classes.
'''

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from transformers.tokenization_utils_base import BatchEncoding
from sentence_transformers import SentenceTransformer

from utils import count_tokens, get_keywords


class Encoder(ABC):
  '''
  Base class for encoders.

  :param tokenizer: Hugging Face tokenizer
  :param int min_tokens: Min tokens in text encodings
  :param int max_tokens: Max tokens in text encodings
  :param optional preprocessor: Text preprocessor
  :param bool = True add_special_tokens: Add BOS and EOS tokens to text before summary generation
  :param int | None = None bos_id: Beginning Of Sentence (BOS) token id
  :param int | None = None eos_id: End Of Sentence (EOS) token id
  '''

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
    self.num_special_tokens = int(bos_id is not None) + int(eos_id is not None)

  def __call__(
    self,
    texts: str | list[str],
    return_batch: bool = True,
    **kwargs
  ) -> list[int] | list[list[int]] | BatchEncoding:
    '''
    Encodes texts to fit in the model's context size and creates a BatchEncoding.

    :param str | list[str] texts: Texts (or text) to encode.
    :param bool = True return_batch: Whether to return a BatchEncoding or not.
    :param **kwargs: Override default `min_tokens` or `max_tokens`.

    :returns encodings (list[int] | list[list[int]] | BatchEncoding):Batched text encodings.
    '''
    preprocessor = self.preprocessor
    min_tokens = kwargs.get('min_tokens', self.min_tokens)
    max_tokens = kwargs.get('max_tokens', self.max_tokens)
    # Preprocess texts
    if preprocessor is not None:
      texts = preprocessor(texts)
    # Convert to list if single text is given
    single_text = isinstance(texts, str)
    if single_text:
      texts = [texts]
    # Encode texts
    encodings = [
      self._encode_wrapper(text, min_tokens, max_tokens)
      for text in texts
    ]
    # Return single encoding if single text is given
    if single_text:
      encodings = encodings[0]
    # Return BatchEncoding if specified
    if return_batch:
      encodings = self.tokenizer.pad(
        {'input_ids': encodings},
        return_tensors = 'pt',
        verbose = False
      )
    return encodings

  @abstractmethod
  def encode(self, text: str, **kwargs) -> list[int]:
    '''
    Creates encoding for a given text with number of tokens in the range [`min_tokens`, `max_tokens`].

    :param str text: Text to encode
    :param **kwargs: Override default `min_tokens` or `max_tokens`

    :returns encoding (list[int]): Text encodings
    '''
    ...

  def _encode_wrapper(
    self,
    text: str,
    min_tokens: int,
    max_tokens: int
  ) -> list[int]:
    '''
    Wrapper for the encode method to handle special tokens and token limits.
    '''
    # Subtract special tokens if they are added
    if self.add_special_tokens:
      max_tokens -= self.num_special_tokens
    # Check if text fits in the model
    num_tokens, encoding = count_tokens(text, self.tokenizer)
    if num_tokens > max_tokens:
      encoding = self.encode(
        text=text,
        min_tokens=min_tokens,
        max_tokens=max_tokens
      )
    # Add special tokens if specified
    if self.add_special_tokens:
      bos_id = self.bos_id
      eos_id = self.eos_id
      if bos_id is not None:
        encoding = [bos_id] + encoding
      if eos_id is not None:
        encoding = encoding + [eos_id]
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
      tokenizer,
      0,
      max_tokens,
      preprocessor,
      add_special_tokens,
      tokenizer.bos_token_id,
      tokenizer.eos_token_id
    )
    self.head_size = head_size

  def encode(self, text: str, **kwargs) -> list[int]:
    tokenizer = self.tokenizer
    max_tokens = kwargs.get('max_tokens', self.max_tokens)
    # Encode the text
    num_tokens, encoding = count_tokens(text, tokenizer)
    # Calculate indices of head and tail
    head_idx = int(max_tokens * self.head_size)
    tail_idx = num_tokens - max_tokens + head_idx
    # Truncate the middle and concatenate head and tail
    encoding = np.concatenate([
      encoding[:head_idx],
      encoding[tail_idx:]
    ]).astype(int).tolist()
    return encoding


class UniformSampler(Encoder):

  def __init__(
    self,
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    text_segmenter: Callable[[str], list[str]],
    preprocessor: Callable[[list[str]], list[str]] | None = None,
    seed: int | None = None,
    segment_delimiter: str = ' ',
    add_special_tokens: bool = True
  ) -> None:
    super().__init__(
      tokenizer,
      min_tokens,
      max_tokens,
      preprocessor,
      add_special_tokens,
      tokenizer.bos_token_id,
      tokenizer.eos_token_id
    )
    self.text_segmenter = text_segmenter
    self.seed = seed
    self.segment_delimiter = segment_delimiter
    np.random.seed(seed)

  def encode(self, text: str, **kwargs) -> list[int]:
    tokenizer = self.tokenizer
    min_tokens = kwargs.get('min_tokens', self.min_tokens)
    max_tokens = kwargs.get('max_tokens', self.max_tokens)
    # Check if encodings fit in the model
    len_encoding, _ = count_tokens(text, tokenizer)
    # Extract and tokenize segments
    segments = self.text_segmenter(text)
    segments = np.array(segments)
    num_segments = len(segments)
    # Approximate probability of picking a segment
    p = max_tokens / len_encoding
    # Sample until segments fit in model
    while True:
      # Create sampling mask
      segment_mask = np.random.rand(num_segments) <= p
      sampled = segments[segment_mask]
      # Flatten and tokenize sampled segments
      flattened = self.segment_delimiter.join(sampled)
      flattened = tokenizer(
        flattened,
        add_special_tokens = False,
        verbose = False
      )['input_ids']
      # Return if number of tokens is in range
      if min_tokens <= len(flattened) <= max_tokens:
        return flattened


class SegmentSampler(Encoder):

  def __init__(
    self,
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    text_segmenter: Callable[[str], list[str]],
    sent_encoder: SentenceTransformer,
    preprocessor: Callable[[list[str]], list[str]] | None = None,
    threshold: float = .7,
    prob_boost: float = .03,
    seed: int | None = None,
    segment_delimiter: str = ' ',
    add_special_tokens: bool = True
  ) -> None:
    super().__init__(
      tokenizer,
      min_tokens,
      max_tokens,
      preprocessor,
      add_special_tokens,
      tokenizer.bos_token_id,
      tokenizer.eos_token_id
    )
    self.text_segmenter = text_segmenter
    self.sent_encoder = sent_encoder
    self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
    self.threshold = threshold
    self.prob_boost = prob_boost
    self.seed = seed
    self.segment_delimiter = segment_delimiter
    np.random.seed(seed)

  def encode(self, text: str, **kwargs) -> list[int]:
    tokenizer = self.tokenizer
    text_segmenter = self.text_segmenter
    sent_encoder = self.sent_encoder
    min_tokens = kwargs.get('min_tokens', self.min_tokens)
    max_tokens = kwargs.get('max_tokens', self.max_tokens)
    # Extract and tokenize segments
    segments = text_segmenter(text)
    # Approximate probability of picking a segment
    num_tokens, _ = count_tokens(text, tokenizer)
    p = (1 + self.prob_boost) * max_tokens / num_tokens
    # Sample until segments fit in model
    while True:
      # Initialize sampled embedding
      num_sampled = 0
      sampled_embedding = np.zeros(self.sent_embedding_dim)
      # Sample segments
      sampled_segments = []
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
        sampled_segments.append(segment)
        # Update sampled embedding
        sampled_embedding = (
          (num_sampled * sampled_embedding + segment_embedding) /
          (num_sampled + 1)
        )
        num_sampled += 1
      # Flatten and tokenize sampled segments
      flattened = self.segment_delimiter.join(sampled_segments)
      flattened = tokenizer(
        flattened,
        add_special_tokens=False,
        verbose=False
      )['input_ids']
      # Return if number of tokens is in range
      if min_tokens <= len(flattened) <= max_tokens:
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
    threshold: float = .7,
    seed: int | None = None,
    segment_delimiter: str = ' ',
    add_special_tokens: bool = True
  ) -> None:

    super().__init__(
      tokenizer,
      min_tokens,
      max_tokens,
      preprocessor,
      add_special_tokens,
      tokenizer.bos_token_id,
      tokenizer.eos_token_id
    )
    self.text_segmenter = text_segmenter
    self.sent_encoder = sent_encoder
    self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
    self.threshold = threshold
    self.seed = seed
    self.segment_delimiter = segment_delimiter
    np.random.seed(seed)

  def encode(self, text: str, **kwargs) -> list[int]:
    tokenizer = self.tokenizer
    segment_delimiter = self.segment_delimiter
    min_tokens = kwargs.get('min_tokens', self.min_tokens)
    max_tokens = kwargs.get('max_tokens', self.max_tokens)
    # Extract segments
    segments = self.text_segmenter(text)
    # Remove redundant segments
    segments = self.remove_redundancy(segments)
    num_segments = len(segments)
    # Count number of tokens in segments
    num_tokens, _ = count_tokens(segments, tokenizer)
    # Account for segment delimiters
    num_tokens += num_segments - 1
    # Check if segments fit in the model
    if num_tokens <= max_tokens:
      flattened = segment_delimiter.join(segments)
      flattened = tokenizer(
        flattened,
        add_special_tokens=False,
        verbose=False
      )['input_ids']
      return flattened
    # Approximate probability of picking a segment
    p = max_tokens / num_tokens
    # Convert list of segments to numpy array for sampling
    segments = np.array(segments)
    # Sample until segments fit in model
    while True:
      segment_mask = np.random.rand(num_segments) <= p
      sampled = segments[segment_mask]
      # Flatten segments
      flattened = segment_delimiter.join(sampled)
      flattened = tokenizer(
        flattened,
        add_special_tokens=False,
        verbose=False
      )['input_ids']
      # Return if number of tokens is in range
      if min_tokens <= len(flattened) <= max_tokens:
        return flattened

  def remove_redundancy(self, segments: list[str]) -> list[str]:
    sent_encoder = self.sent_encoder
    segment_embs = sent_encoder.encode(segments)
    # Average embedding of selected segments
    selected_emb = np.zeros(self.sent_embedding_dim)
    num_segments = 0
    selected_segments = []
    for segment, embedding in zip(segments, segment_embs):
      # Calculate similarity between current segment and chosen segments
      similarity = selected_emb @ embedding
      # Discard current segment and contnue if it is similar
      if self.threshold < similarity:
        continue
      # Otherwise select it
      selected_segments.append(segment)
      # Update selected segments embedding
      selected_emb = (num_segments * selected_emb + embedding) / (num_segments + 1)
      num_segments += 1
    return selected_segments


class RemoveRedundancy2(Encoder):

  def __init__(
    self,
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    text_segmenter: Callable[[str], list[str]],
    sent_encoder: SentenceTransformer,
    preprocessor: Callable[[list[str]], list[str]] | None = None,
    threshold: float = .7,
    seed: int | None = None,
    segment_delimiter: str = ' ',
    add_special_tokens: bool = True
  ) -> None:
    super().__init__(
      tokenizer,
      min_tokens,
      max_tokens,
      preprocessor,
      add_special_tokens,
      tokenizer.bos_token_id,
      tokenizer.eos_token_id
    )
    self.text_segmenter = text_segmenter
    self.sent_encoder = sent_encoder
    self.sent_embedding_dim = sent_encoder.get_sentence_embedding_dimension()
    self.threshold = threshold
    self.seed = seed
    self.segment_delimiter = segment_delimiter
    np.random.seed(seed)

  def encode(self, text: str, **kwargs) -> list[int]:
    tokenizer = self.tokenizer
    segment_delimiter = self.segment_delimiter
    min_tokens = kwargs.get('min_tokens', self.min_tokens)
    max_tokens = kwargs.get('max_tokens', self.max_tokens)
    # Extract segments
    segments = self.text_segmenter(text)
    # Get keywords
    keywords = get_keywords(text)
    # Convert list of segments to numpy array for sampling
    segments = np.array(segments)
    # Remove redundant segments
    segments = self.remove_redundancy(segments, keywords)
    num_segments = len(segments)
    # Count number of tokens in segments
    num_tokens, _ = count_tokens(segments, tokenizer)
    # Account for segment delimiters
    num_tokens += num_segments - 1
    # Check if segments fit in the model
    if num_tokens <= max_tokens:
      flattened = segment_delimiter.join(segments)
      flattened = tokenizer(
        flattened,
        add_special_tokens=False,
        verbose=False
      )['input_ids']
      return flattened
    # Approximate probability of picking a segment
    p = max_tokens / num_tokens
    # Sample until segments fit in model
    while True:
      segment_mask = np.random.rand(num_segments) <= p
      sampled = segments[segment_mask]
      # Flatten segments
      flattened = segment_delimiter.join(sampled)
      flattened = tokenizer(
        flattened,
        add_special_tokens = False,
        verbose = False
      )['input_ids']
      # Return if number of tokens is in range
      if min_tokens <= len(flattened) <= max_tokens:
        return flattened

  def remove_redundancy(
    self,
    segments: np.ndarray[str],
    keywords: list[str]
  ) -> list[str]:
    sent_encoder = self.sent_encoder
    # Get keyword embedding
    keywords = ' '.join(keywords)
    keyword_emb = sent_encoder.encode(keywords)
    # Get segment embeddings
    segment_embs = sent_encoder.encode(segments)
    # Create filter for segments
    scores = segment_embs @ keyword_emb
    filt = scores > self.threshold
    selected_segments = segments[filt]
    return selected_segments


class KeywordScorer(Encoder):

  def __init__(
    self,
    tokenizer,
    max_tokens: int,
    text_segmenter: Callable[[str], list[str]],
    sent_encoder: SentenceTransformer,
    preprocessor: Callable[[list[str]], list[str]] | None = None,
    num_keywords: int = 20,
    keywords_preprocessor: Callable[[list[str]], list[str]] | None = None,
    stop_words: list[str] | None = None,
    segment_delimiter: str = ' ',
    add_special_tokens: bool = True
  ) -> None:
    super().__init__(
      tokenizer,
      0,
      max_tokens,
      preprocessor,
      add_special_tokens,
      tokenizer.bos_token_id,
      tokenizer.eos_token_id
    )
    self.text_segmenter = text_segmenter
    self.sent_encoder = sent_encoder
    self.num_keywords = num_keywords
    self.keywords_preprocessor = keywords_preprocessor
    self.stop_words = stop_words
    self.segment_delimiter = segment_delimiter

  def encode(self, text: str, **kwargs) -> list[str]:
    tokenizer = self.tokenizer
    max_tokens = kwargs.get('max_tokens', self.max_tokens)
    sent_encoder = self.sent_encoder
    # Extract keywords from the text
    keywords = get_keywords(
      text, self.num_keywords, self.stop_words,
      self.keywords_preprocessor
    )
    # Create keywords embedding
    keywords_emb = sent_encoder.encode(' '.join(keywords))
    # Extract segments from the text
    segments = self.text_segmenter(text)
    # Get segment embeddings
    segment_embeddings = sent_encoder.encode(segments)
    # Calculate similarity of keywords with each segment
    segment_similarities = segment_embeddings @ keywords_emb
    # Argument sort the similarities
    best_indices = np.argsort(segment_similarities)[::-1]
    # Select maximum segment indices with highest scores
    selected_indices = []
    tokens_used = 0
    for i in best_indices:
      segment_len, _ = count_tokens(segments[i], tokenizer)
      if tokens_used + segment_len + 1 > max_tokens:
        continue
      selected_indices.append(i)
      # +1 to account for segment delimiter
      tokens_used += segment_len + 1
    # Sort the selected indices to maintain segment order
    selected_indices.sort()
    # Flatten and tokenize selected segments
    flattened = self.segment_delimiter.join([
      segments[i] for i in selected_indices
    ])
    flattened = tokenizer(
      flattened,
      add_special_tokens=False,
      verbose=False
    )['input_ids']
    return flattened

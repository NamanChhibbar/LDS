import re
from typing import Callable

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


inf = float("inf")



def count_words(text: str) -> int:
	return len(text.split())

def count_tokens(
	texts: str | list[str],
	tokenizer
) -> int:
	if isinstance(texts, str):
		texts = [texts]
	encodings = tokenizer(
		texts, add_special_tokens=False
	)["input_ids"]
	num_tokens = np.sum([
		len(encoding) for encoding in encodings
	])
	return num_tokens

def get_device() -> str:
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"

def show_exception(exc: Exception) -> None:
	exc_class = exc.__class__.__name__
	exc_msg = str(exc)
	print(f"Encountered exception of type {exc_class}: {exc_msg}")

def clear_stdout(spaces: int = 100) -> None:
	print(f"\r{" " * spaces}\r", end="")

def get_keywords(
	text: str,
	num_words: int = 20,
	stop_words: list[str] | None = None,
	preprocessor: Callable[[str], str] | None = None
) -> list[str]:
	vectorizer = CountVectorizer(
		stop_words=stop_words,
		preprocessor=preprocessor
	)
	dtm = vectorizer.fit_transform([text])
	lda = LatentDirichletAllocation(n_components=1)
	lda.fit(dtm)
	feature_names = vectorizer.get_feature_names_out()
	topic_words = [
		feature_names[i]
		for i in lda.components_[0].argsort()[:-num_words-1:-1]
	]
	return topic_words



class TextProcessor:

	# Matches everything except words, numbers, and single quotes
	_non_word_pat_sub = (r"[^\w\s']", "")

	_preprocessing_pats_subs = [
		# Non-ASCII quotes
		(r"‘|’", "'"),
		(r"“|”", '"'),
		# Non-ASCII characters
		(r"[^\x00-\x7f]+", ""),
		# Emails
		(r"\S+@\S+\.\S+", ""),
		# Hyperlinks
		(r"\S+://\S+\.\S+", ""),
		# Hashtags
		(r"#\S+", ""),
		# HTML tags
		(r"<[^\n>]+>", ""),
		# Remove unecessary periods
		(r"\.\s*([,;:?!-])", r"\1"),
		(r"([,;:?!-])\s*\.", r"\1"),
		(r"\.\s+([a-z])", r" \1"),
		# Remove ending period of abbreviations
		# (due to difficulties in sentence segmentation)
		(r"(\w\.\w+)\.(\s)", r"\1\2"),
		# Remove unecessary decimal points
		(r"(\d+)\.(\s)", r"\1\2"),
		# Repair punctuations
		(r"(\w)\s+([,.;:?!-])", r"\1\2"),
		# Join broken sentences
		(r"(\w[,;]?)\s+(\w)", r"\1 \2"),
	]
	# Remove numbers
	_number_pat_sub = (r"(\b|\+)[\d+-]+\b", "")

	# White spaces
	_whitespace_pats_subs = [
		# Replace multiple spaces and tabs
		(r"[ \t]+", " "),
		# Remove spaces around newline
		(r" ?\n ?", "\n"),
		# Replace multiple newlines
		(r"\n{3,}", "\n\n"),
	]

	def __init__(
			self,
			only_words_nums: bool = False,
			preprocessing: bool = False,
			remove_nums: bool = False,
			ignore_tokens: list[str] | None = None
		) -> None:
		pats_subs = []

		# Only words and numbers
		if only_words_nums:
			pats_subs.append(TextProcessor._non_word_pat_sub)

		# Preprocessing
		if preprocessing:
			pats_subs.extend(TextProcessor._preprocessing_pats_subs)

		# Remove numbers
		if remove_nums:
			pats_subs.append(TextProcessor._number_pat_sub)

		# Ignore specific tokens
		if ignore_tokens is not None:
			pats_subs.append((re.compile(r"|".join(ignore_tokens)), ""))

		# Fix white spaces
		pats_subs.extend(TextProcessor._whitespace_pats_subs)

		self.pats_subs = [
			(re.compile(pat), sub) for pat, sub in pats_subs
		]
	
	def __call__(
		self,
		texts: str | list[str]
	) -> str | list[str]:
		if isinstance(texts, str):
			return self.process(texts)
		texts = [self.process(text) for text in texts]
		return texts
		
	def process(
		self,
		text: str
	) -> str:
		for pat, sub in self.pats_subs:
			text = pat.sub(sub, text)
		text = text.strip()
		return text



class TextSegmenter:

	def __init__(
		self,
		base_tokenizer: Callable[[str], list[str]],
		min_words: int,
		sent_delimiter: str = " "
	) -> None:
		self.base_tokenizer = base_tokenizer
		self.min_words = min_words
		self.sent_delimiter = sent_delimiter
	
	def __call__(
		self,
		text: str
	) -> list[str]:
		min_words = self.min_words
		sent_delimiter = self.sent_delimiter
		parts = self.base_tokenizer(text)
		num_parts = len(parts)
		segments = []
		for i, sent in enumerate(parts):
			if count_words(sent) >= min_words:
				segments.append(sent)
				continue
			prev_text_words = count_words(segments[-1]) if \
				segments else inf
			next_text_words = count_words(parts[i+1]) if \
				i + 1 < num_parts else inf
			if prev_text_words < next_text_words:
				segments[-1] = f"{segments[-1]}{sent_delimiter}{sent}"
			else:
				parts[i + 1] = f"{sent}{sent_delimiter}{parts[i+1]}"
		return segments

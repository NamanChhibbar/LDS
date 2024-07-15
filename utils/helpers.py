import re
from typing import Callable

import numpy as np
import torch


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



class TextProcessor:

	# Basic preprocessing
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
		# Replace multiple spaces and tabs
		(r"[ \t]+", " "),
		# Remove spaces around newline
		(r" ?\n ?", "\n"),
		# Replace multiple newlines
		(r"\n{3,}", "\n\n"),
	]
	# Remove numbers
	_number_pat_sub = (r"[+?\d+-?]+", "")

	def __init__(
			self,
			preprocessing: bool = False,
			remove_nums: bool = False,
			ignore_tokens: list[str] | None = None
		) -> None:
		pats_subs = []

		# Ignore specific tokens
		if ignore_tokens is not None:
			pats_subs.append((re.compile(r"|".join(ignore_tokens)), ""))

		# Include preprocessing patterns
		if preprocessing:
			pats_subs.extend(TextProcessor._preprocessing_pats_subs)

		# Include numbers removal
		if remove_nums:
			pats_subs.append(TextProcessor._number_pat_sub)

		self.pats_subs = [
			(re.compile(pat), sub) for pat, sub in pats_subs
		]
	
	def __call__(
		self,
		texts: str | list[str]
	) -> list[str]:
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
			if next_text_words < prev_text_words:
				parts[i + 1] = f"{sent}{sent_delimiter}{parts[i+1]}"
			else:
				segments[-1] = f"{segments[-1]}{sent_delimiter}{sent}"
		return segments

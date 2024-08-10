import re
import typing
import subprocess

import numpy as np
import torch
import sklearn
import nltk


inf = float("inf")



def gpu_usage() -> list[int]:
	"""
	Get the current GPU memory usage.
	"""
	# Get output from nvidia-smi
	result = subprocess.check_output([
		"nvidia-smi",
		"--query-gpu=memory.used",
		"--format=csv,nounits,noheader"
	]).decode("utf-8").strip()

	# Extract memory used by GPUs in MiB
	gpu_memory = [int(mem) for mem in result.split("\n")]

	return gpu_memory


def get_device(threshold: int | float = 500) -> str:
	"""
	Returns a device with memory usage below `threshold`.
	"""
	if torch.cuda.is_available():
		usage = gpu_usage()
		cuda_ind = np.argmin(usage)
		return f"cuda:{cuda_ind}" if usage[cuda_ind] < threshold \
			else "cpu"
	if torch.backends.mps.is_available():
		usage = torch.mps.driver_allocated_memory() / 1e6
		return "mps" if usage < threshold else "cpu"
	return "cpu"


def count_words(text: str) -> int:
	words = text.split()
	num_words = len(words)
	return num_words


def count_tokens(
	texts: str | list[str],
	tokenizer
) -> tuple[int, list[int]] | tuple[int, list[list[int]]]:
	encodings = tokenizer(
		texts,
		add_special_tokens = False,
		verbose = False
	)["input_ids"]
	num_tokens = len(encodings) if isinstance(texts, str) \
		else sum([len(encoding) for encoding in encodings])
	return num_tokens, encodings


def show_exception(exception: Exception) -> None:
	exc_class = exception.__class__.__name__
	exc_msg = str(exception)
	print(f"\nEncountered exception of type {exc_class}: {exc_msg}\n")


def clear_stdout(spaces: int = 100) -> None:
	print(f"\r{" " * spaces}", end="\r")


def get_keywords(
	text: str,
	num_words: int = 20,
	stop_words: list[str] | None = None,
	preprocessor: typing.Callable[[str], str] | None = None
) -> list[str]:
	vectorizer = sklearn.feature_extraction.text.CountVectorizer(
		stop_words = stop_words,
		preprocessor = preprocessor
	)
	dtm = vectorizer.fit_transform([text])
	lda = sklearn.decomposition.LatentDirichletAllocation(n_components=1)
	lda.fit(dtm)
	feature_names = vectorizer.get_feature_names_out()
	topic_words = [
		feature_names[i]
		for i in lda.components_[0].argsort()[:-num_words-1:-1]
	]
	return topic_words


def get_stop_words(
	extra_stop_words: list[str] | None = None,
	lang: str = "english"
) -> list[str]:
	stop_words = nltk.corpus.stopwords.words(lang)
	if extra_stop_words is not None:
		stop_words += [
			word.lower()
			for word in extra_stop_words
			if word not in stop_words
		]
	stop_words += [
		word.capitalize()
		for word in stop_words
		if not word.istitle()
	]
	return stop_words



class TextProcessor:

	# Matches everything except words, numbers, and single quotes
	_non_word_pat_sub = (r"[^\w\s']", "")

	_preprocessing_pats_subs = [
		# Remove hyperlinks
		(r"https?://[^\s]+", ""),
		# Remove unecessary periods
		(r"\.\s*([,;:?!-])", r"\1"),
		(r"([,;:?!-])\s*\.", r"\1"),
		(r"\.(\s+)([a-z])", r"\1\2"),
		# Remove ending period of abbreviations
		# (due to difficulties in sentence segmentation)
		(r"(\.\w+)\.(\s)", r"\1\2"),
		# Fix spaces before and after punctuations
		(r"\s+([,.;:?!])", r"\1"),
		(r",([^\s\d])", r", \1"),
		(r"([;:?!])(\S)", r"\1 \2"),
		# Remove spaces within brackets and quotes
		(
			r'"([^"]*)"',
			lambda m: f'"{m.group(1).strip()}"'
		), (
			r"'([^']*)'",
			lambda m: f"'{m.group(1).strip()}'"
		), (
			r"“([^”]*)”",
			lambda m: f"“{m.group(1).strip()}”"
		), (
			r"‘([^’]*)’",
			lambda m: f"‘{m.group(1).strip()}’"
		), (
			r"\[([^\]]*)\]",
			lambda m: f"[{m.group(1).strip()}]"
		), (
			r"\(([^\)]*)\)",
			lambda m: f"({m.group(1).strip()})"
		),
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

		self._pats_subs = [
			(re.compile(pat), sub) for pat, sub in pats_subs
		]
	
	def __call__(
		self,
		texts: str | list[str]
	) -> str | list[str]:	

		# Check if single text is given		
		single_text = isinstance(texts, str)
		if single_text:
			texts = [texts]
		
		# Process texts
		processed_texts = []
		for text in texts:
			for pat, sub in self._pats_subs:
				text = pat.sub(sub, text)
			text = text.strip()
			processed_texts.append(text)

		return processed_texts[0] if single_text else processed_texts



class TextSegmenter:

	def __init__(
		self,
		base_tokenizer: typing.Callable[[str], list[str]],
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
		new_parts = []
		for part in parts:
			new_parts.extend(part.split(";"))
		parts = new_parts
		num_parts = len(parts)

		segments = []
		for i, sent in enumerate(parts):
			prev_text_words = (
				count_words(segments[-1]) if
				segments else inf
			)
			next_text_words = (
				count_words(parts[i + 1]) if
				i + 1 < num_parts else inf
			)
			if (
				count_words(sent) >= min_words or
				prev_text_words == next_text_words == inf
			):
				segments.append(sent)
			elif prev_text_words < next_text_words:
				segments[-1] = f"{segments[-1]}{sent_delimiter}{sent}"
			else:
				parts[i + 1] = f"{sent}{sent_delimiter}{parts[i + 1]}"

		return segments

import re

import torch



def count_words(text: str):
	return len(text.split())

def get_device() -> str:
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"

def extract_special_tokens(token_list):
	all_tokens = []
	for token in token_list:
		all_tokens += token if isinstance(token, list) else [token]
	return all_tokens

def show_exception(exc: Exception):
	exc_class = exc.__class__.__name__
	exc_msg = str(exc)
	print(f"Encountered exception of type {exc_class}: {exc_msg}")



class TextProcessor:

	_preprocessing_pats_subs = [
		# Non-ASCII quotes
		(r"‘|’", "'"),
		(r"“|”", '"'),
		# Non-ASCII characters
		(r"[^\x00-\x7f]+", ""),
		# Emails
		(r"[^\s]+@[^\s]+\.com", ""),
		# Hyperlinks
		(r"[^\s]*://[^\s]*", ""),
		# Hashtags
		(r"#[^\s]+", ""),
		# HTML tags
		(r"<[^\n>]+>", ""),
		# Remove hanging periods
		(r"(\s)+\.(\s)+", r"\1\2"),
		# Remove unecessary periods
		(r"\.\s*([;:\?-])", r"\1"),
		# Remove ending period of abbreviations
		# (due to difficulties in sentence segmentation)
		(r"(\w+.\w+)\.", r"\1"),
		# Remove unecessary decimal points
		(r"(\d+)\.(\s)", r"\1\2")
	]

	# Numbers
	_number_pat_sub = (r"[+?\d+-?]+", "")

	_whitespace_pats_subs = [
		# Multiple spaces and tabs
		(r"([ \t]){2,}", r"\1"),
		# Spaces and tabs before newline
		(r"[ \t]\n", "\n"),
		# Multiple newlines
		(r"\n{3,}", "\n\n"),
	]

	def __init__(
			self, preprocessing: bool=False, remove_nums: bool=False,
			ignore_tokens: list[str]|None=None
		) -> None:
		pats_subs = []
		if preprocessing:
			pats_subs.extend(TextProcessor._preprocessing_pats_subs)
		if remove_nums:
			pats_subs.append(TextProcessor._number_pat_sub)
		if ignore_tokens is not None:
			pats_subs.append((re.compile(r"|".join(ignore_tokens)), ""))
		pats_subs.extend(TextProcessor._whitespace_pats_subs)
		self.pats_subs = [
			(re.compile(pat), sub) for pat, sub in pats_subs
		]
	
	def __call__(self, texts: str|list[str]) -> list[str]:
		if isinstance(texts, str):
			texts = [texts]
		texts = [self.process(text) for text in texts]
		return texts
		
	def process(self, text: str) -> str:
		for pat, sub in self.pats_subs:
			text = pat.sub(sub, text)
		text = text.strip()
		return text

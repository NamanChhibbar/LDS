import re
import torch


class TextProcessor:

	preprocessing_pats_subs = [
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
		(r"<[^\n>]+>", "")
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
			self, pats_subs: list[tuple[str]]=[], ignore_tokens: list[str]=[],
			remove_nums: bool=False
		):
		if remove_nums:
			pats_subs.append(TextProcessor._number_pat_sub)
		if ignore_tokens:
			pats_subs.append((re.compile(r"|".join(ignore_tokens)), ""))
		pats_subs.extend(TextProcessor._whitespace_pats_subs)
		self.pats_subs = [
			(re.compile(pat), sub) for pat, sub in pats_subs
		]
	
	def __call__(self, texts: list[str]):
		if isinstance(texts, str):
			texts = [texts]
		texts = [self.process(text) for text in texts]
		return texts
		
	def process(self, text: str):
		for pat, sub in self.pats_subs:
			text = pat.sub(sub, text)
		text = text.strip()
		return text


def get_device():
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def max_lengths(model):
	model_configs = model.config.to_dict()
	max_input = model_configs["max_position_embeddings"]
	max_output = model_configs["max_length"]
	return max_input, max_output


def count_words(text):
	return len(text.split())


def combine_subsections(sections):
	text = ""
	for sec in sections:
		sec_text = "\n\n".join(sec["paragraphs"])
		if sec["section_title"]:
			sec_text = f"Section {sec["section_title"]}:\n\n{sec_text}"
		text = f"{text}\n\n{sec_text}" if text else sec_text
		if sec["subsections"]:
			sub_text = combine_subsections(sec["subsections"])
			text = f"{text}\n\n{sub_text}" if text else sub_text
	return text

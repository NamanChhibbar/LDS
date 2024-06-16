import re
import numpy as np
import torch

def count_words(text):
	return len(text.split())

class TextPreprocessor:

	def __init__(self, stop_words=None):
		# Match non-ASCII quotes
		self.single_quote = re.compile(r"‘|’")
		self.double_quote = re.compile(r"“|”")
		# Match non-ASCII characters
		self.non_ascii = re.compile(r"[^\x00-\x7f]+")
		# Match emails
		self.email = re.compile(r"[^\s]+@[^\s]+\.com")
		# Match hyperlinks
		self.hyperlink = re.compile(r"[^\s]*://[^\s]*")
		# Match hashtags
		self.hashtag = re.compile(r"#[^\s]+")
		# Match HTML tags
		self.html = re.compile(r"<[^\n>]+>")
		# Match numbers
		self.number = re.compile(r"[+?\d+-?]+")
		# Match stop words
		self.stop_words = re.compile(r"|".join([
			rf"\W?{word}(\W)" for word in stop_words
		])) if stop_words else None
		# Match multiple spaces and tabs
		self.spaces_tabs = re.compile(r"([ \t]){2,}")
		# Match spaces and tabs before newline
		self.space_before_newline = re.compile(r"[ \t]\n")
		# Match multiple newlines
		self.newlines = re.compile(r"\n{3,}")

	def __call__(self, texts: list[str], remove_numbers=False):
		if isinstance(texts, str):
			texts = [texts]
		for i, text in enumerate(texts):
			texts[i] = self.preprocess(text, remove_numbers)
		return texts

	def preprocess(self, text: str, remove_numbers=False):
		# Convert non-ASCII quotes to ASCII quotes
		text = self.single_quote.sub("'", text)
		text = self.double_quote.sub('"', text)
		# Remove non-ASCII characters
		text = self.non_ascii.sub("", text)
		# Remove emails
		text = self.email.sub("", text)
		# Remove hyperlinks
		text = self.hyperlink.sub("", text)
		# Remove hashtags
		text = self.hashtag.sub("", text)
		# Remove HTML tags
		text = self.html.sub("", text)
		# Remove numbers
		if remove_numbers:
			text = self.number.sub("", text)
		# Remove stop words
		if self.stop_words:
			text = self.stop_words.sub(r"\1", text)
		# Concatenate multiple spaces and tabs
		text = self.spaces_tabs.sub(r"\1", text)
		# Remove spaces and tabs before newline
		text = self.space_before_newline.sub("\n", text)
		# Concatenate multiple newlines
		text = self.newlines.sub("\n\n", text)
		# Remove trailing and leading spaces
		text = text.strip()
		return text

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

def max_lengths(model):
	configs = model.config.to_dict()
	max_input = configs["max_position_embeddings"]
	max_output = configs["max_length"]
	return max_input , max_output

def pick_sents(text, sent_tokenizer, tokenizer, context_size):
	sents = sent_tokenizer(text)
	sents = tokenizer(sents)["input_ids"]
	mean_length = np.mean([
		len(sent) for sent in sents
	])
	num_samples = int(context_size / mean_length)
	sents = np.array(sents, dtype=object)
	while True:
		sampled = np.random.choice(sents, size=num_samples, replace=False)
		flattened = np.array([
			elm for lis in sampled for elm in lis
		])
		if len(flattened) <= context_size:
			return torch.tensor(flattened, dtype=int)

def truncate_middle(texts, tokenizer, size, head_size=.5):
	head_idx = int(size * head_size)
	truncated_ids = []
	for text in texts:
		text_ids = tokenizer.encode(text)
		tail_idx = len(text_ids) - size + head_idx
		truncated = np.concatenate([
			text_ids[:head_idx],
			text_ids[tail_idx:]
		])
		truncated_ids.append(truncated)
	padded_ids = tokenizer.pad({
		"input_ids": truncated_ids
		}, return_tensors="pt")
	return padded_ids

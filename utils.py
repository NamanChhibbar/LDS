import re
import numpy as np
import torch

def count_words(text):
	return len(text.split())

def preprocess_text(text, stop_words=None):

	# Convert non-ASCII quotes to ASCII quotes
	text = re.sub(r"‘|’", "'", text)
	text = re.sub(r"“|”", '"', text)

	# Remove non-ASCII characters
	text = re.sub(r"[^\x00-\x7f]+", "", text)

	# Remove emails
	text = re.sub(r"[^\s]+@[^\s]+\.com", "", text)

	# Remove hyperlinks
	text = re.sub(r"[^\s]*://[^\s]*", "", text)

	# Remove hashtags
	text = re.sub(r"#[^\s]+", "", text)

	# Remove HTML tags
	text = re.sub(r"<[^\n>]+>", "", text)

	# Remove numbers
	# text = re.sub(r"[+?\d+-?]+", "", text)

	# Remove stop words
	if stop_words:
		text = re.sub(r"|".join([
			rf"\W?{word}(\W)" for word in stop_words
		]), r"\1", text)

	# Concatenating multiple spaces and tabs
	text = re.sub(r"([ \t]){2,}", r"\1", text)

	# Removing spaces and tabs before newline
	text = re.sub(r"[ \t]\n", "\n", text)

	# Concatenating multiple newlines
	text = re.sub(r"\n{3,}", "\n\n", text)

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

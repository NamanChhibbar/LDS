import re
import numpy as np
import torch


class TextPreprocessor:

	def __init__(self, stop_words=None, remove_nums=False):
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
		self.number = re.compile(r"[+?\d+-?]+") if remove_nums else None
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

	def __call__(self, texts: list[str]):
		if isinstance(texts, str):
			texts = [texts]
		for i, text in enumerate(texts):
			texts[i] = self.preprocess(text)
		return texts

	def preprocess(self, text: str):
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
		if self.number:
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


class TextPostprocessor:

	def __init__(self, special_tokens: list[str]):
		self.special_tokens = re.compile(r"|".join(special_tokens))
	
	def __call__(self, texts: list[str]):
		if isinstance(texts, str):
			texts = [texts]
		texts = [self.special_tokens.sub("", text) for text in texts]
		return texts


class SummarizationPipeline:

	def __init__(self, text_preprocessor, text_postprocessor):
		self.preprocessor = text_preprocessor
		self.postprocessor = text_postprocessor

	def __call__(self, texts: list[str]):
		if isinstance(texts, str):
			texts = [texts]
		preprocessed = self.preprocessor(texts)
		summaries = self.summarize(preprocessed)
		postprocessed = self.postprocessor(summaries)
		return postprocessed
	
	def summarize(self, texts: list[str]):
		...


class UniformSampler(SummarizationPipeline):

	def __init__(
			self, text_preprocessor, text_postprocessor,  sent_tokenizer, tokenizer,
			summarizer, summarizer_context_size, max_output_tokens, device="cpu"
		):
		super().__init__(text_preprocessor, text_postprocessor)
		self.sent_tokenizer = sent_tokenizer
		self.tokenizer = tokenizer
		self.summarizer = summarizer.to(device)
		self.context_size = summarizer_context_size
		self.max_tokens = max_output_tokens
		self.device = device
	
	def summarize(self, texts: list[str]):
		inputs = self.pick_sents(texts).to(self.device)
		outputs = self.summarizer.generate(**inputs, max_length=self.max_tokens)
		summaries = [self.tokenizer.decode(out) for out in outputs]
		return summaries

	def pick_sents(self, texts):
		sent_tokenizer = self.sent_tokenizer
		tokenizer = self.tokenizer
		context_size = self.context_size

		processed_texts = []
		for text in texts:
			# Extract and encode sentences
			sents = sent_tokenizer(text)
			sents = tokenizer(sents)["input_ids"]
			sents = np.array(sents, dtype=list)

			# Mean length of sentences
			mean_length = np.mean([
				len(sent) for sent in sents
			])

			# Approximate number of sentences needed
			num_samples = int(context_size / mean_length)

			# Check if there are enough sentences
			if len(sents) <= num_samples:
				flattened = [elm for lis in sents for elm in lis]
				processed_texts.append(flattened)
				continue

			# Sample until sentences fit in model
			while True:
				sampled = np.random.choice(sents, size=num_samples, replace=False)
				flattened = [elm for lis in sampled for elm in lis]
				if len(flattened) <= context_size:
					processed_texts.append(flattened)
					break

		# Pad sentences and create attention mask
		padded_ids = tokenizer.pad({
			"input_ids": processed_texts
		}, return_tensors="pt")

		return padded_ids


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


def truncate_middle(texts, tokenizer, size, head_size=.5):
	# Constant head size
	head_size = int(size * head_size)
	truncated_ids = []

	for text in texts:
		# Encode the text
		text_ids = tokenizer.encode(text)

		# Check if ids fit in model
		if len(text_ids) <= size:
			truncated_ids.append(text_ids)
			continue

		# Calculate beginning index of tail
		tail_idx = len(text_ids) - size + head_size

		# Truncate the middle and concatenate head and tail
		truncated = np.concatenate([
			text_ids[:head_size],
			text_ids[tail_idx:]
		])
		truncated_ids.append(truncated)
	
	# Pad sentences and create attention mask
	padded_ids = tokenizer.pad({
		"input_ids": truncated_ids
		}, return_tensors="pt")

	return padded_ids

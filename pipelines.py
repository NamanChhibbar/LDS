import numpy as np
from abc import ABC, abstractmethod


class SummarizationPipeline(ABC):

	def __init__(
			self, text_preprocessor, text_postprocessor, tokenizer, summarizer,
			max_output_tokens: int, device="cpu"
		):
		self.preprocessor = text_preprocessor
		self.postprocessor = text_postprocessor
		self.tokenizer = tokenizer
		self.summarizer = summarizer.to(device)
		self.max_tokens = max_output_tokens
		self.device = device

	def __call__(self, texts: list[str]):
		if isinstance(texts, str):
			texts = [texts]
		preprocessed = self.preprocessor(texts)
		inputs = self.generate_ids(preprocessed).to(self.device)
		outputs = self.summarizer.generate(**inputs, max_length=self.max_tokens)
		summaries = [self.tokenizer.decode(out) for out in outputs]
		postprocessed = self.postprocessor(summaries)
		return postprocessed
	
	@abstractmethod
	def generate_ids(self, texts: list[str]):
		...
	

class TruncateMiddle(SummarizationPipeline):

	def __init__(
			self, text_preprocessor, text_postprocessor, tokenizer, summarizer,
			context_size: int, max_output_tokens: int, head_size: float=.5,
			device="cpu"
		):
		super().__init__(
			text_preprocessor, text_postprocessor, tokenizer, summarizer,
			max_output_tokens, device
		)
		self.context_size = context_size
		self.head_size = head_size

	def generate_ids(self, texts: list[str]):
		# Constant head size
		head_size = int((size := self.context_size) * self.head_size)
		truncated_ids = []

		for text in texts:
			# Encode the text
			text_ids = self.tokenizer.encode(text)

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
		padded_ids = self.tokenizer.pad({
			"input_ids": truncated_ids
			}, return_tensors="pt")

		return padded_ids


class UniformSampler(SummarizationPipeline):

	def __init__(
			self, text_preprocessor, text_postprocessor,  sent_tokenizer, tokenizer,
			summarizer, summarizer_context_size: int, max_output_tokens: int,
			device="cpu"
		):
		super().__init__(
			text_preprocessor, text_postprocessor, tokenizer, summarizer,
			max_output_tokens, device
		)
		self.sent_tokenizer = sent_tokenizer
		self.context_size = summarizer_context_size

	def generate_ids(self, texts: list[str]):
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

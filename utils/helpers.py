import re
from time import perf_counter
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.tokenization_utils_base import BatchEncoding
from bert_score import BERTScorer

from .pipelines import Encoder


def count_words(text: str):
	return len(text.split())


def get_device() -> str:
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


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
			self, preprocessing: bool=False, remove_nums: bool=False,
			ignore_tokens: list[str]=None
		) -> None:
		pats_subs = []
		if preprocessing:
			pats_subs.extend(TextProcessor._preprocessing_pats_subs)
		if remove_nums:
			pats_subs.append(TextProcessor._number_pat_sub)
		if ignore_tokens:
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


class SummarizationDataset:

	def __init__(
			self, texts_summaries: list[tuple[str]], encoder: Encoder,
			batch_size: int, context_size: int, use_cache: bool=False,
			shuffle: bool=False, seed: int|None=None
		) -> None:
		# This enables dynamic batching
		texts_summaries = sorted(
			texts_summaries, key=lambda x: count_words(x[0])
		)
		num_texts = len(texts_summaries)
		self.num_batches = num_texts // batch_size

		# Storing batches of (text, summary) in a numpy array
		self.text_batches = np.zeros(self.num_batches, dtype=object)
		for i in range(self.num_batches):
			batch = texts_summaries[i*batch_size:(i+1)*batch_size]
			self.text_batches[i] = batch

		# Using cache as a numpy array, if specified
		self.cached = np.zeros(
			self.num_batches, dtype=object
		) if use_cache else None

		self.encoder = encoder
		self.batch_size = batch_size
		self.context_size = context_size
		self.shuffle = shuffle
		self.seed = seed
		np.random.seed(seed)
		self.it = None
	
	def __iter__(self):
		self.it = 0
		if self.shuffle:
			permutation = np.random.permutation(self.num_batches)
			self.text_batches = self.text_batches[permutation]
			if self.cached is not None:
				self.cached = self.cached[permutation]
		return self
	
	def __next__(self) -> BatchEncoding:
		# Check if iterator is not implemented or if iterations are completed
		if self.it is None or self.it == self.num_batches:
			raise StopIteration()
		it = self.it
		self.it += 1

		# Check if input is cached
		cached = self.cached
		if cached is not None and cached[it]:
			return cached[it]
		
		# Encode texts using encoder and summaries using tokenizer
		tokenizer = self.encoder.tokenizer
		texts_summaries = self.text_batches[it]
		texts = [pair[0] for pair in texts_summaries]
		summaries = [pair[1] for pair in texts_summaries]
		text_encodings = self.encoder(texts)
		summ_encodings = tokenizer(
			summaries, padding=True, max_length=self.context_size,
			truncation=True, return_tensors="pt"
		)["input_ids"]

		# Set padding token ids to -100 (ignored id in attention)
		filt = summ_encodings == tokenizer.pad_token_id
		summ_encodings[filt] = -100

		# Create batch encoding
		batch_encodings = BatchEncoding({
			**text_encodings, "labels": summ_encodings
		})

		# Save to cache and delete text bacth if using cache
		if cached is not None:
			cached[it] = batch_encodings
			self.text_batches[it] = 0

		return batch_encodings


class Evaluator:

	def __init__(
			self, pipelines, texts_summaries: tuple[str]|list[tuple[str]],
			device: str|torch.device|None=None
		) -> None:
		if not isinstance(texts_summaries, list):
			texts_summaries = [texts_summaries]
		self.pipelines = pipelines
		self.texts = [pair[0] for pair in texts_summaries]
		self.summaries = [pair[1] for pair in texts_summaries]
		self.bert_scorer = BERTScorer(lang="en", device=device)
		self.generated_summaries = []
	
	def generate_summaries(self) -> list[int]:
		summaries = self.generated_summaries
		time_taken = []
		for pipeline in self.pipelines:
			start = perf_counter()
			summary = pipeline(self.texts)
			time = (perf_counter() - start) * 1000
			summaries.extend(summary)
			time_taken.append(time)
		return time_taken

	def get_bertscore(self) -> list[torch.Tensor]:
		if not self.generated_summaries:
			print("Generating summaries")
			self.generate_summaries()
		summaries = self.summaries
		num_pipelines = len(self.pipelines)
		summaries *= num_pipelines
		metrics = self.bert_scorer.score(self.generated_summaries, summaries)
		metrics = [
			metric.reshape((num_pipelines, -1)).mean(dim=1)
			for metric in metrics
		]
		return metrics


def train_model(
	model, dataset: SummarizationDataset, epochs: int,
	optimizer: Optimizer, scheduler: LRScheduler=None,
	device: str|torch.device|None=None, flt_prec: int=4
) -> list[int]:
	model = model.to(device)
	epoch_losses = []
	num_batches = len(dataset)

	model.train(True)
	for epoch in range(epochs):
		epoch_time = 0
		epoch_loss = 0

		for batch, inputs in enumerate(dataset):
			inputs = inputs.to(device)

			start = perf_counter()
			loss = model(**inputs).loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			time = (perf_counter() - start) * 1000

			epoch_time += time
			epoch_loss += loss.item()

			time_remaining = (
				epoch_time * (num_batches * (epochs - epoch) / (batch + 1) - 1)
			) // 1000
			seconds = time_remaining % 60
			minutes = (time_remaining // 60) % 60
			hours = time_remaining // 3600

			print(
				"\r"
				f"Epoch: {epoch+1}/{epochs} "
				f"Batch: {batch+1}/{num_batches} "
				f"Time: {round(time, flt_prec)} ms/batch "
				f"Loss: {round(loss.item(), flt_prec)} "
				f"Time remaining: {hours}h {minutes}m {seconds}s",
				end=""
			)

		epoch_time = epoch_time / num_batches
		epoch_loss = epoch_loss / num_batches
		epoch_losses.append(epoch_loss)

		if scheduler:
			scheduler.step(epoch_loss)

		print(
			f"\rEpoch {epoch+1}/{epochs} "
			f"Avergage time {epoch_time} ms/batch "
			f"Average loss {epoch_loss}"
		)
	return epoch_losses

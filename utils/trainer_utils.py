from math import ceil
from time import perf_counter

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.tokenization_utils_base import BatchEncoding

from .helpers import count_words, show_exception, clear_stdout
from .encoders import Encoder



class SummarizationDataset:
	"""
	Creates an iterable batched dataset of text (and summary) encodings.

	## Parameters
	`texts`: List of texts
	`encoder`: Encoder used to encode `texts`
	`batch_size`: Maximum number of text encodings in a batch
	`summaries`: List of summaries
	`summary_max_tokens`: Maximum tokens in summary encodings
	`use_cache`: Use a cache to store already processed batches while iterating
	`shuffle`: Shuffle batches before iterating
	`seed`: Manual seed for output reproducibility
	"""
	def __init__(
		self,
		texts: list[str],
		encoder: Encoder,
		batch_size: int,
		summaries: list[str] | None = None,
		summary_max_tokens: int = 0,
		use_cache: bool = False,
		shuffle: bool = False,
		seed: int | None = None
	) -> None:
		# This enables dynamic batching
		perm = np.argsort([count_words(text) for text in texts])
		texts = np.array(texts)[perm]
		if summaries is not None:
			summaries = np.array(summaries)[perm]

		# Store batches of texts and summaries in a numpy array
		num_batches = self.num_batches = ceil(len(texts) / batch_size)
		self.text_batches = np.zeros(num_batches, dtype=object)
		self.summary_batches = None if summaries is None else \
			np.zeros(num_batches, dtype=object)
		for i in range(self.num_batches):
			text_batch = texts[i*batch_size:(i+1)*batch_size].tolist()
			self.text_batches[i] = text_batch
			if summaries is not None:
				summary_batch = summaries[i*batch_size:(i+1)*batch_size].tolist()
				self.summary_batches[i] = summary_batch

		# Use numpy array as a cache, if specified
		self.cached = np.zeros(
			self.num_batches, dtype=object
		) if use_cache else None

		self.encoder = encoder
		self.batch_size = batch_size
		self.summary_max_tokens = summary_max_tokens
		self.shuffle = shuffle
		self.seed = seed
		np.random.seed(seed)
		self.it = None

	def __len__(self):
		return self.num_batches
	
	def __getitem__(
		self, ind: int
	) -> BatchEncoding:
		encoder = self.encoder
		cached = self.cached
		# Check if input is cached
		if cached is not None and cached[ind]:
			return cached[ind]
		
		# Encode texts using encoder and summaries using tokenizer
		text_batches = self.text_batches
		summary_batches = self.summary_batches
		texts = text_batches[ind]
		encodings = encoder(texts)
		if summary_batches is not None:
			tokenizer = encoder.tokenizer
			summaries = summary_batches[ind]
			summ_encodings = tokenizer(
				summaries, padding=True, max_length=self.summary_max_tokens,
				truncation=True, return_tensors="pt"
			)["input_ids"]

			# Set padding token ids to -100 (ignored id in attention)
			filt = summ_encodings == tokenizer.pad_token_id
			summ_encodings[filt] = -100
			encodings["labels"] = summ_encodings

		# Create batch encoding
		batch_encodings = BatchEncoding(encodings)

		# Save to cache and delete text batch if using cache
		if cached is not None:
			cached[ind] = batch_encodings
			text_batches[ind] = 0
			if summary_batches is not None:
				summary_batches[ind] = 0

		return batch_encodings

	def __iter__(self):
		self.it = 0
		if self.shuffle:
			permutation = np.random.permutation(self.num_batches)
			self.text_batches = self.text_batches[permutation]
			if self.summary_batches is not None:
				self.summary_batches = self.summary_batches[permutation]
			if self.cached is not None:
				self.cached = self.cached[permutation]
		return self
	
	def __next__(self) -> BatchEncoding:
		it = self.it
		# Check if iterator is initialized
		assert it is not None, "Iterator not initialized"
		# Check if iterations are completed
		if it == self.num_batches:
			raise StopIteration()
		self.it += 1
		return self[it]
	


def train_model(
	model,
	dataset: SummarizationDataset,
	epochs: int,
	optimizer: Optimizer,
	scheduler: LRScheduler | None = None,
	device: str | torch.device = "cpu",
	flt_prec: int = 4
) -> list[int]:

	# For clearing output
	SPACES = 100

	model = model.to(device)
	epoch_losses = []
	num_batches = len(dataset)

	model.train(True)
	for epoch in range(epochs):
		# Track total epoch loss and time
		epoch_loss = 0
		epoch_time = 0

		for batch, inputs in enumerate(dataset):
			start = perf_counter()
			try:
				inputs = inputs.to(device)
				loss = model(**inputs).loss
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			except Exception as e:
				show_exception(e)
				print("Training terminated")
				model.train(False)
				return epoch_losses

			epoch_loss += loss.item()
			time = (perf_counter() - start) * 1000

			epoch_time += time

			# Calculate remaining time
			seconds = int(
				epoch_time * (num_batches * (epochs - epoch) / (batch + 1) - 1)
			) // 1000
			minutes = seconds // 60
			hours = minutes // 60
			days = hours // 24

			time_remaining = f"{seconds % 60}s"
			if minutes:
				time_remaining = f"{minutes % 60}m {time_remaining}"
			if hours:
				time_remaining = f"{hours % 24}h {time_remaining}"
			if days:
				time_remaining = f"{days}d {time_remaining}"

			clear_stdout(SPACES)
			print(
				f"Epoch [{epoch+1}/{epochs}] "
				f"Batch [{batch+1}/{num_batches}] "
				f"Time [{round(time, flt_prec)} ms/batch] "
				f"Loss [{round(loss.item(), flt_prec)}] "
				f"Time remaining [{time_remaining}]",
				end=""
			)

		epoch_loss = epoch_loss / num_batches
		epoch_time = epoch_time / num_batches
		epoch_losses.append(epoch_loss)

		if scheduler is not None:
			scheduler.step(epoch_loss)

		clear_stdout(SPACES)
		print(
			f"\rEpoch [{epoch+1}/{epochs}] "
			f"Average loss [{round(epoch_loss, flt_prec)}] "
			f"Avergage time [{round(epoch_time, flt_prec)} ms/batch]"
		)
	model.train(False)
	return epoch_losses

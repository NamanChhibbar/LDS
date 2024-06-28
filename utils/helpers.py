from math import ceil
from os import cpu_count
import re
from time import perf_counter
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.tokenization_utils_base import BatchEncoding
from bert_score import BERTScorer
from rouge import Rouge



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



class Encoder(ABC):
	"""
	Base class for encoders
	"""
	def __init__(
			self, tokenizer, preprocessor: TextProcessor|None=None,
			bos_id: int|None=None, eos_id: int|None=None
		) -> None:
		"""
		## Parameters
		`tokenizer`: Hugging Face tokenizer
		`preprocessor`: Text preprocessor
		"""
		super().__init__()
		self.tokenizer = tokenizer
		self.preprocessor = preprocessor
		self.bos_id = bos_id
		self.eos_id = eos_id

	def __call__(self, texts: str|list[str]) -> BatchEncoding:
		"""
		Encode texts

		## Parameters
		`texts`: Texts (or text) to encode

		## Returns
		`encodings`: Text encodings of type BatchEncoding
		"""
		if isinstance(texts, str):
			texts = [texts]
		if self.preprocessor is not None:
			texts = self.preprocessor(texts)
		encodings = self.generate_encodings(texts)
		return encodings
	
	@abstractmethod
	def generate_encodings(self, texts: list[str]) -> BatchEncoding:
		...
	
	def add_special_tokens(self, encodings: list[int]):
		bos_id = self.bos_id
		eos_id = self.eos_id
		if bos_id is not None:
			text_ids = [bos_id] + encodings
		if eos_id is not None:
			text_ids = text_ids + [eos_id]
		return text_ids



class SummarizationDataset:

	def __init__(
		self, texts: list[str], encoder: Encoder, batch_size: int,
		summaries: list[str]|None=None, context_size: int|None=None,
		use_cache: bool=False, shuffle: bool=False, seed: int|None=None
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

		# Use cache as a numpy array, if specified
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

	def __len__(self):
		return self.num_batches
	
	def __getitem__(self, ind):
		# Check if input is cached
		cached = self.cached
		if cached is not None and cached[ind]:
			return cached[ind]
		
		# Encode texts using encoder and summaries using tokenizer
		text_batches = self.text_batches
		summary_batches = self.summary_batches
		texts = text_batches[ind]
		encodings = self.encoder(texts)
		if summary_batches is not None:
			tokenizer = self.encoder.tokenizer
			summaries = summary_batches[ind]
			summ_encodings = tokenizer(
				summaries, padding=True, max_length=self.context_size,
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
		# Check if iterator is not implemented or if iterations are completed
		it = self.it
		if it is None or it == self.num_batches:
			raise StopIteration()
		self.it += 1
		return self[it]



class Evaluator:

	def __init__(
			self, pipelines, texts: str|list[str], summaries: str|list[str],
			rouge_metrics: list[str]|None=None, rougen_max_n: int=2,
			rougew_weight_factor: int=1.2, device: str|torch.device|None=None
		) -> None:
		if isinstance(texts, str):
			texts = [texts]
		if isinstance(summaries, str):
			summaries = [summaries]

		# Initialize pipelines, texts, and summaries
		self.pipelines = pipelines
		self.num_pipelines = len(pipelines)
		self.texts = texts
		self.summaries = summaries

		# Initialise ROUGE scorer
		if rouge_metrics is None:
			rouge_metrics = ["rouge-n", "rouge-l", "rouge-w"]
		self.rouge_scorer = Rouge(
			metrics=rouge_metrics, max_n=rougen_max_n, limit_length=False,
			weight_factor=rougew_weight_factor
		)
		if "rouge-n" in rouge_metrics:
			rouge_metrics.remove("rouge-n")
			self.rouge_metrics = [
				f"rouge-{i+1}" for i in range(rougen_max_n)
			]
			self.rouge_metrics.extend(rouge_metrics)
		else:
			self.rouge_metrics = rouge_metrics
		self.rougen_max_n = rougen_max_n
		self.rougew_weight_factor = rougew_weight_factor

		# Initialize BERT scorer
		self.bert_scorer = BERTScorer(lang="en", device=device)
		self.device = device
		self.generated_summaries = None
	
	def generate_summaries(
		self, batch_size: int|None=None, num_workers: int|None=None
	) -> list[int]:
		summaries = self.generated_summaries = []
		time_taken = []
		inputs = [
			(i, batch_size) for i in range(self.num_pipelines)
		]
		if num_workers is not None and num_workers > 1:
			with ProcessPoolExecutor(max_workers=num_workers) as executor:
				results = executor.map(self._generate_summaries, inputs)
		else:
			results = map(self._generate_summaries, inputs)
		for summary, time in results:
			summaries.extend(summary)
			time_taken.append(time)
		return time_taken
	
	# F, P, R
	def get_rouge_score(self) -> list[dict[str, np.ndarray]]:
		generated_summaries = self.generated_summaries
		assert generated_summaries is not None, "Summaries not generated"
		num_generated_summaries = len(generated_summaries)
		summaries = self.summaries
		num_summaries = len(summaries)
		scores = []
		for i in range(0, num_generated_summaries, num_summaries):
			pipeline_summaries = generated_summaries[i:i+num_summaries]
			mean_score = {
				metric: np.array([0., 0, 0])
				for metric in self.rouge_metrics
			}
			for cand, ref in zip(pipeline_summaries, summaries):
				score = self.rouge_scorer.get_scores(cand, ref)
				for metric, values in score.items():
					mean_score[metric] += list(values.values())
			for metric, values in mean_score.items():
				mean_score[metric] = (values / num_summaries).tolist()
			scores.append(mean_score)
		return scores

	# P, R, F
	def get_bert_score(self) -> list[torch.Tensor]:
		generated_summaries = self.generated_summaries
		assert generated_summaries is not None, "Summaries not generated"
		num_pipelines = self.num_pipelines
		summaries = num_pipelines * self.summaries
		metrics = self.bert_scorer.score(generated_summaries, summaries)
		metrics = np.array([
			metric.reshape((num_pipelines, -1)).mean(dim=1)
			for metric in metrics
		])
		order = [2, 0, 1]
		metrics = metrics.T[:, order].tolist()
		return metrics
	
	def _generate_summaries(self, args):
		ind, batch_size = args
		pipeline = self.pipelines[ind]
		start = perf_counter()
		summaries = pipeline(self.texts, batch_size)
		time_taken = (perf_counter() - start)
		print(f"Generated summary for pipeline {ind} in {time_taken}s")
		return summaries, time_taken



def train_model(
	model, dataset: SummarizationDataset, epochs: int,
	optimizer: Optimizer, scheduler: LRScheduler|None=None,
	device: str|torch.device|None=None, flt_prec: int=4
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
				print(
					f"Encountered exception of type {type(e)}: {e}\n"
					"Training terminated"
				)
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

			
			print(
				f"\r{" " * SPACES}\r"
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

		print(
			f"\r{" " * SPACES}\r"
			f"\rEpoch [{epoch+1}/{epochs}] "
			f"Average loss [{round(epoch_loss, flt_prec)}] "
			f"Avergage time [{round(epoch_time, flt_prec)} ms/batch]"
		)
	return epoch_losses

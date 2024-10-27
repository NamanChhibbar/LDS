"""
Contains callable end-to-end summarization pipelines.
"""

import time
import abc
from collections.abc import Callable

import torch
import openai

from encoders import Encoder
from utils import count_tokens, show_exception, SummarizationDataset



class Pipeline(abc.ABC):

	def __init__(
		self,
		model,
		encoder: Encoder,
		postprocessor: Callable[[list[str]], list[str]] | None = None
	) -> None:

		self.model = model
		self.encoder = encoder
		self.postprocessor = postprocessor

	def __call__(
		self,
		texts: str | list[str],
		**kwargs
	) -> str | list[str]:

		return self.generate_summaries([texts], **kwargs)[0] \
			if isinstance(texts, str) else \
			self.generate_summaries(texts, **kwargs)

	@abc.abstractmethod
	def generate_summaries(
		self,
		texts: list[str],
		**kwargs
	) -> list[str]:
		...



class SummarizationPipeline(Pipeline):
	"""
	Pipeline for generating summaries using an encoder.

	:param model: The model model.
	:param Encoder encoder: The encoder model.
	:param int | None = None summary_min_tokens: The minimum number of tokens in the summary.
	:param int | None = None summary_max_tokens: The maximum number of tokens in the summary.
	:param (list[str]) -> list[str] | None = None postprocessor: The postprocessor for the generated summaries.
	:param str | torch.device = "cpu" device: The device to use for computation.
	:param float = 1.0 temperature: The temperature for sampling.
	:param float = 1.0 repetition_penalty: The repetition penalty.
	:param float = 0.9 top_p: The nucleus sampling threshold.

	## Returns
	list[str]: The generated summaries.
	"""

	def __init__(
		self,
		model,
		encoder: Encoder,
		postprocessor: Callable[[list[str]], list[str]] | None = None,
		summary_min_tokens: int | None = None,
		summary_max_tokens: int | None = None,
		device: str | torch.device = "cpu",
		temperature: float = 1.,
		repetition_penalty: float = 1.,
		top_p: float = .9
	) -> None:

		super().__init__(model.to("cpu"), encoder, postprocessor)
		self.summary_min_tokens = summary_min_tokens or model.config.min_length
		self.summary_max_tokens = summary_max_tokens or encoder.max_tokens
		self.device = device
		self.temperature = temperature
		self.repetition_penalty = repetition_penalty
		self.top_p = top_p

	def generate_summaries(
		self,
		texts: list[str],
		**kwargs
	) -> list[str]:

		device = self.device
		model = self.model.to(device)
		encoder = self.encoder
		postprocessor = self.postprocessor
		summary_min_tokens = self.summary_min_tokens
		summary_max_tokens = self.summary_max_tokens
		batch_size = kwargs.get("batch_size", 1)
		temperature = kwargs.get("temperature", self.temperature)
		repetition_penalty = kwargs.get(
			"repetition_penalty", self.repetition_penalty
		)
		top_p = kwargs.get("top_p", self.top_p)

		# Generate encodings in batches
		batches = SummarizationDataset(texts, encoder, batch_size)

		# Generate summaries
		all_summaries = []
		for encodings in batches:

			# Send encodings to device
			encodings = encodings.to(device)

			# Generate summaries' encodings
			output = model.generate(
				**encodings,
				min_length = summary_min_tokens,
				max_length = summary_max_tokens,
				temperature = temperature,
				repetition_penalty = repetition_penalty,
				top_p = top_p,
				early_stopping = True
			)

			# Decode summaries' encodings
			summaries = [
				encoder.tokenizer.decode(out, skip_special_tokens=True)
				for out in output
			]

			# Append summaries
			all_summaries.extend(summaries)

		# Remove model from device
		model.to("cpu")

		# Postprocess summaries
		if postprocessor is not None:
			all_summaries = postprocessor(all_summaries)

		return all_summaries



class OpenAIPipeline(Pipeline):

	def __init__(
		self,
		model: str,
		encoder: Encoder,
		postprocessor: Callable[[list[str]], list[str]] | None = None,
		system_prompt: str | None = None,
		delay: float = 1.
	) -> None:

		super().__init__(model, encoder, postprocessor)
		self.max_tokens = encoder.max_tokens
		self.system_prompt = system_prompt
		self.delay = delay
		self.call_inputs = None
		self.response = None
	
	def generate_summaries(
		self,
		texts: list[str],
		**_
	) -> list[str]:

		postprocessor = self.postprocessor

		summaries = []
		for text in texts:

			# Create call inputs
			self.create_inputs(text)

			# Extract summary if call is successful
			summary = ""
			if self.send_call():
				summary = self.response.choices[0].message.content

			# Postprocess summary
			if postprocessor is not None:
				summary = postprocessor(summary)

			# Append summary
			summaries.append(summary)

			# Delay before next call
			time.sleep(self.delay)

		return summaries
	
	def create_inputs(
		self,
		text: str
	) -> int:

		encoder = self.encoder
		max_tokens = self.max_tokens
		tokenizer = encoder.tokenizer

		# Tokens used to create OpenAI prompt template
		# 3 tokens for prompt base
		# 4 tokens each for every message
		tokens_used = 7

		# Create system prompt
		system_prompt = self.system_prompt
		messages = []
		if system_prompt is not None:
			messages.append({"role": "system", "content": system_prompt})
			tokens_used += count_tokens(system_prompt, tokenizer)[0] + 4

		# Distill text
		encodings = encoder(
			text,
			return_batch = False,
			max_tokens = max_tokens - tokens_used
		)
		text = tokenizer.decode(encodings, ignore_special_tokens=True)

		# Create prompt
		messages.append({"role": "user", "content": text})

		# Create inptuts
		self.call_inputs = {
			"model": self.model,
			"messages": messages,
			"max_tokens": max_tokens
		}
		return tokens_used
	
	def send_call(self) -> bool:

		# Check if call inputs are created
		call_inputs = self.call_inputs
		assert call_inputs is not None, "Call inputs not created"

		# Send call
		try:
			self.response = openai.chat.completions.create(**call_inputs)

		# Show exception and return False if the call failed
		except Exception as e:
			show_exception(e)
			return False

		# Return True if call is successful
		return True

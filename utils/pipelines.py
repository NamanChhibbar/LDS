from time import sleep
from abc import ABC, abstractmethod
from typing import Callable

import torch
import openai

from .helpers import count_tokens, show_exception
from .encoders import Encoder
from .trainer_utils import SummarizationDataset


OPENAI_DELAY = 3



class Pipeline(ABC):

	def __init__(
		self,
		model,
		encoder: Encoder,
		postprocessor: Callable[[list[str]], list[str]] | None = None
	) -> None:
		self.model = model
		self.encoder = encoder
		self.postprocessor = postprocessor

	@abstractmethod
	def __call__(
		self,
		texts: str | list[str],
		batch_size: int | None = None
	) -> list[str]:
		pass



class SummarizationPipeline(Pipeline):
	"""
	Pipeline for generating summaries using an encoder.

	## Parameters
	`model`: The model model.
	`encoder`: The encoder model.
	`summary_min_tokens`: The minimum number of tokens in the summary.
	`summary_max_tokens`: The maximum number of tokens in the summary.
	`postprocessor`: The postprocessor for the generated summaries.
	`device`: The device to use for computation.

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
		device: str | torch.device = "cpu"
	) -> None:
		super().__init__(model.to("cpu"), encoder, postprocessor)
		self.summary_min_tokens = summary_min_tokens or model.config.min_length
		self.summary_max_tokens = summary_max_tokens or encoder.max_tokens
		self.device = device

	def __call__(
		self,
		texts: str | list[str],
		batch_size: int | None = None
	) -> list[str]:
		if isinstance(texts, str):
			texts = [texts]
		
		device = self.device
		model = self.model.to(device)
		encoder = self.encoder
		summary_max_tokens = self.summary_max_tokens
		postprocessor = self.postprocessor
		batch_size = batch_size or len(texts)

		# Generate encodings in batches
		batches = SummarizationDataset(texts, encoder, batch_size)

		# Generate summaries
		all_summaries = []
		for encoding in batches:
			# Send encodings to device
			encoding = encoding.to(device)

			# Generate summaries' encodings
			output = self.model.generate(
				**encoding, max_length=summary_max_tokens
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
		system_prompt: str | None = None
	) -> None:
		super().__init__(model, encoder, postprocessor)
		self.max_tokens = encoder.max_tokens
		self.system_prompt = system_prompt
		self.call_inputs = None
		self.response = None
	
	def __call__(
		self,
		texts: list[str],
		_ = None
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
			sleep(OPENAI_DELAY)

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
			tokens_used += count_tokens(system_prompt, tokenizer) + 4

		# Distill text
		encodings = encoder.encode(
			text,
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

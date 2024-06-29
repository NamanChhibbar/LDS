import torch
import openai

from .helpers import TextProcessor, count_tokens, show_exception
from .encoders import Encoder
from .trainer_utils import SummarizationDataset



class SummarizationPipeline:

	def __init__(
			self, summarizer, encoder: Encoder, max_tokens: int,
			postprocessor: TextProcessor|None=None,
			device: str|torch.device|None=None
		) -> None:
		self.summarizer = summarizer.to(device)
		self.encoder = encoder
		self.max_tokens = max_tokens
		self.postprocessor = postprocessor
		self.device = device

	def __call__(
			self, texts: str|list[str], batch_size: int|None=None
		) -> list[str]:
		if isinstance(texts, str):
			texts = [texts]
		if batch_size is None:
			batch_size = len(texts)
		encoder = self.encoder
		dataset = SummarizationDataset(texts, encoder, batch_size)
		summaries = []
		for encodings in dataset:
			encodings = encodings.to(self.device)
			outputs = self.summarizer.generate(
				**encodings, max_length=self.max_tokens
			)
			summaries.extend([
				encoder.tokenizer.decode(out, skip_special_tokens=True)
				for out in outputs
			])
		if self.postprocessor is not None:
			summaries = self.postprocessor(summaries)
		return summaries



class OpenAIPipeline:

	def __init__(
		self, model: str, encoder: Encoder,
		prompt_template: str="", system_prompt: str=""
	) -> None:
		self.model = model
		self.encoder = encoder
		self.max_tokens = encoder.max_tokens
		self.prompt_template = prompt_template
		self.system_prompt = system_prompt
		self.call_inputs = None
		self.response = None
	
	def __call__(self):
		...
	
	def create_inputs(
		self, text: str, previous_messages: list[str]|None=None
	) -> int:
		encoder = self.encoder
		max_tokens = self.max_tokens
		prompt_template = self.prompt_template
		tokenizer = encoder.tokenizer

		# Tokens used to create OpenAI prompt template
		# 3 tokens for prompt base
		# 4 tokens each for every message
		num_prev_msgs = 0 if previous_messages is None else len(previous_messages)
		tokens_used = 3 + 4 * (2 * num_prev_msgs + 1)

		# Create system prompt
		system_prompt = self.system_prompt
		messages = []
		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})
			tokens_used += count_tokens(system_prompt, tokenizer) + 4
		if num_prev_msgs:
			for text, summary in previous_messages:
				messages.append({"role": "user", "content": text})
				messages.append({"role": "assistant", "content": summary})
				tokens_used += count_tokens([text, summary], tokenizer)
		tokens_used += count_tokens(prompt_template, tokenizer)
		encodings = encoder.encode(text, max_tokens - tokens_used)
		text = tokenizer.decode(encodings, ignore_special_tokens=True)
		prompt = f"{prompt_template}{text}"
		messages.append({"role": "user", "content": prompt})
		self.call_inputs = {
			"model": self.model,
			"messages": messages,
			"max_tokens": max_tokens
		}
		return tokens_used
	
	def send_call(self):
		call_inputs = self.call_inputs
		assert call_inputs is not None, "Call inputs not created"
		try:
			self.response = openai.chat.completions.create(**call_inputs)
		except Exception as e:
			show_exception(e)
		return self.response

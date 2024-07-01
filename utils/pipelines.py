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
		encoder = self.encoder
		postprocessor = self.postprocessor

		if isinstance(texts, str):
			texts = [texts]

		# Create dataset
		# Create a single batch if batch size is None
		batch_size = len(texts) if batch_size is None else batch_size
		dataset = SummarizationDataset(texts, encoder, batch_size)

		summaries = []
		for encodings in dataset:
			encodings = encodings.to(self.device)

			# Generate summaries' encodings
			outputs = self.summarizer.generate(
				**encodings, max_length=self.max_tokens
			)

			# Decode summaries' encodings
			generated_summaries = [
				encoder.tokenizer.decode(out, skip_special_tokens=True)
				for out in outputs
			]

			# Add generated summaries
			summaries.extend(generated_summaries)

		# Postprocess summaries
		if postprocessor is not None:
			summaries = postprocessor(summaries)

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
	
	def __call__(
		self, texts: list[str], prev_msgs: list[str]|None=None
	) -> list[str]:
		summaries = []

		for text in texts:
			# Create call inputs
			self.create_inputs(text, prev_msgs)

			# Return summaries is call is not successful
			if not self.send_call():
				return summaries
			
			# Extract and append summary
			summary = self.response.choices[0].message.content
			summaries.append(summary)

		return summaries
	
	def create_inputs(
		self, text: str, prev_msgs: list[str]|None=None
	) -> int:
		encoder = self.encoder
		max_tokens = self.max_tokens
		prompt_template = self.prompt_template
		tokenizer = encoder.tokenizer

		# Tokens used to create OpenAI prompt template
		# 3 tokens for prompt base
		# 4 tokens each for every message
		num_prev_msgs = 0 if prev_msgs is None else len(prev_msgs)
		tokens_used = 3 + 4 * (2 * num_prev_msgs + 1)

		# Create system prompt
		system_prompt = self.system_prompt
		messages = []
		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})
			tokens_used += count_tokens(system_prompt, tokenizer) + 4

		# Add previous messages, if any
		if num_prev_msgs:
			for text, summary in prev_msgs:
				messages.append({"role": "user", "content": text})
				messages.append({"role": "assistant", "content": summary})
				tokens_used += count_tokens([text, summary], tokenizer)

		# Count tokens in prompt template
		tokens_used += count_tokens(prompt_template, tokenizer)

		# Distill document
		encodings = encoder.encode(text, max_tokens - tokens_used)
		text = tokenizer.decode(encodings, ignore_special_tokens=True)

		# Add prompt to template
		prompt = f"{prompt_template}{text}"
		messages.append({"role": "user", "content": prompt})

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

		try:
			# Send call
			self.response = openai.chat.completions.create(**call_inputs)
		except Exception as e:
			# Show exception and return False if the call failed
			show_exception(e)
			return False

		# Return True if call is successful
		return True

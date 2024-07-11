import torch
import openai

from .helpers import TextProcessor, count_tokens, show_exception
from .encoders import Encoder
from .trainer_utils import SummarizationDataset



class SummarizationPipeline:
	"""
	Pipeline for generating summaries using a summarizer and an encoder.

	## Parameters
	`summarizer`: The summarizer model.
	`encoder`: The encoder model.
	`summary_min_tokens`: The minimum number of tokens in the summary.
	`summary_max_tokens`: The maximum number of tokens in the summary.
	`postprocessor`: The postprocessor for the generated summaries.
	`device`: The device to use for computation.

	## Returns
	list[str]: The generated summaries.
	"""
	def __init__(
		self, summarizer, encoder:Encoder, summary_min_tokens:int|None=None,
		summary_max_tokens:int|None=None, postprocessor:TextProcessor|None=None,
		device:str|torch.device="cpu"
	) -> None:
		self.summarizer = summarizer.to("cpu")
		self.encoder = encoder
		self.summary_min_tokens = summarizer.config.min_length \
			if summary_min_tokens is None else summary_min_tokens
		self.summary_max_tokens = encoder.max_tokens \
			if summary_max_tokens is None else summary_max_tokens
		self.postprocessor = postprocessor
		self.device = device

	def __call__(
		self, texts:str|list[str], batch_size:int|None=None
	) -> list[str]:
		if isinstance(texts, str):
			texts = [texts]
		
		device = self.device
		summarizer = self.summarizer.to(device)
		encoder = self.encoder
		summary_max_tokens = self.summary_max_tokens
		postprocessor = self.postprocessor
		if batch_size is None:
			batch_size = len(texts)

		# Generate encodings in batches
		batches = SummarizationDataset(texts, encoder, batch_size)

		# Generate summaries
		all_summaries = []
		for encoding in batches:

			# Send encodings to device
			encoding = encoding.to(device)

			# Generate summaries' encodings
			output = self.summarizer.generate(
				**encoding, max_length=summary_max_tokens
			)

			# Decode summaries' encodings
			summaries = [
				encoder.tokenizer.decode(out, skip_special_tokens=True)
				for out in output
			]

			# Append summaries
			all_summaries.extend(summaries)

		# Remove summarizer from device
		summarizer.to("cpu")

		# Postprocess summaries
		if postprocessor is not None:
			all_summaries = postprocessor(all_summaries)

		return all_summaries



class OpenAIPipeline:

	def __init__(
		self, model:str, encoder:Encoder,
		prompt_template:str="", system_prompt:str=""
	) -> None:
		self.model = model
		self.encoder = encoder
		self.max_tokens = encoder.max_tokens
		self.prompt_template = prompt_template
		self.system_prompt = system_prompt
		self.call_inputs = None
		self.response = None
	
	def __call__(self, texts:list[str], _=None) -> list[str]:
		summaries = []

		for text in texts:
			# Create call inputs
			self.create_inputs(text)

			# Return summaries is call is not successful
			if not self.send_call():
				summary = ""
			
			# Extract and append summary
			summary = self.response.choices[0].message.content
			summaries.append(summary)

		return summaries
	
	def create_inputs(self, text:str) -> int:
		encoder = self.encoder
		max_tokens = self.max_tokens
		prompt_template = self.prompt_template
		tokenizer = encoder.tokenizer

		# Tokens used to create OpenAI prompt template
		# 3 tokens for prompt base
		# 4 tokens each for every message
		tokens_used = 7

		# Create system prompt
		system_prompt = self.system_prompt
		messages = []
		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})
			tokens_used += count_tokens(system_prompt, tokenizer) + 4

		# Count tokens in prompt template
		tokens_used += count_tokens(prompt_template, tokenizer)

		# Distill text
		encodings = encoder.encode(text, max_tokens - tokens_used)
		text = tokenizer.decode(encodings, ignore_special_tokens=True)

		# Create prompt
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

		# Send call
		try:
			self.response = openai.chat.completions.create(**call_inputs)

		# Show exception and return False if the call failed
		except Exception as e:
			show_exception(e)
			return False

		# Return True if call is successful
		return True

import torch

from .helpers import TextProcessor
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

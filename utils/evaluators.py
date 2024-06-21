import torch
from bert_score import BERTScorer


class Evaluator:

	def __init__(
			self, pipelines, texts: str|list[str], summaries: str|list[str],
			device: str|torch.device|None=None
		):
		if len(texts) != len(summaries):
			raise ValueError("Number of texts and summaries differ")
		self.pipelines = pipelines
		self.texts = texts if isinstance(texts, list) else [texts]
		self.summaries = summaries if isinstance(summaries, list) else [summaries]
		self.bert_scorer = BERTScorer(lang="en", device=device)
		self.generated_summaries = []
	
	def generate_summaries(self):
		summaries = self.generated_summaries
		for pipeline in self.pipelines:
			summary = pipeline(self.texts)
			summaries.extend(summary)

	def get_bertscore(self):
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

from time import perf_counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from bert_score import BERTScorer
from rouge import Rouge



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
	
	def _generate_summaries(self, args):
		ind, batch_size = args
		pipeline = self.pipelines[ind]
		start = perf_counter()
		summaries = pipeline(self.texts, batch_size)
		time_taken = (perf_counter() - start)
		print(f"Generated summary for pipeline {ind} in {time_taken}s")
		return summaries, time_taken

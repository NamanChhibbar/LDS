from time import perf_counter

import numpy as np
import torch
from bert_score import BERTScorer
from rouge import Rouge

from .pipelines import Pipeline



class Evaluator:

	def __init__(
		self,
		pipelines: list[Pipeline],
		device: str | torch.device = "cpu",
		rouge_metrics: list[str] | None = None,
		rougen_max_n: int = 2,
		rougew_weight_factor: int = 1.2
	) -> None:
		# Initialize pipelines
		pipelines = self.pipelines = pipelines if \
			isinstance(pipelines, list) else [pipelines]
		self.num_pipelines = len(pipelines)

		# Initialize BERT scorer
		self.bert_scorer = BERTScorer(lang="en", device=device)
		self.device = device

		# Initialise ROUGE scorer
		rouge_metrics = rouge_metrics or ["rouge-n", "rouge-l", "rouge-w"]
		self.rouge_scorer = Rouge(
			metrics=rouge_metrics,
			max_n=rougen_max_n,
			limit_length=False,
			weight_factor=rougew_weight_factor
		)
		if "rouge-n" in rouge_metrics:
			rouge_metrics.remove("rouge-n")
			rouge_metrics = [
				f"rouge-{i + 1}"
				for i in range(rougen_max_n)
			] + rouge_metrics
		self.rouge_metrics = rouge_metrics
		self.rougen_max_n = rougen_max_n
		self.rougew_weight_factor = rougew_weight_factor

		self.summaries = None

	def __call__(
		self,
		texts: str | list[str],
		summaries: str | list[str],
		batch_size: int | None = None
	) -> dict:
		time_taken = self.generate_summaries(texts, batch_size)
		bert_score = self.get_bert_score(summaries)
		rouge_score = self.get_rouge_score(summaries)
		scores = {
			"time-taken": time_taken,
			"bert-scores": bert_score,
			"rouge-scores": rouge_score
		}
		return scores

	def generate_summaries(
		self,
		texts: str | list[str],
		batch_size: int | None = None
	) -> list[int]:
		if isinstance(texts, str):
			texts = [texts]
		all_summaries = self.summaries = []
		time_taken = []
		for i, pipeline in enumerate(self.pipelines):
			print(f"Generating summaries for pipeline {i + 1}...")
			start = perf_counter()
			summaries = pipeline(texts, batch_size=batch_size)
			time = perf_counter() - start
			print(f"Took {time}s")
			all_summaries.extend(summaries)
			time_taken.append(time)
		return time_taken
	
	# P, R, F
	def get_bert_score(
		self,
		summaries: str | list[str]
	) -> list[list[float]]:
		all_summaries = self.summaries
		assert all_summaries is not None, "Summaries not generated"
		num_pipelines = self.num_pipelines
		summaries = num_pipelines * summaries
		metrics = self.bert_scorer.score(all_summaries, summaries)
		metrics = np.array([
			metric.reshape((num_pipelines, -1)).mean(dim=1)
			for metric in metrics
		])
		order = [2, 0, 1]
		metrics *= 100
		metrics = metrics.T[:, order].tolist()
		return metrics
	
	# F, P, R
	def get_rouge_score(
		self,
		summaries: str | list[str]
	) -> list[dict[str, list[float]]]:
		generated_summaries = self.summaries
		assert generated_summaries is not None, "Summaries not generated"
		num_generated_summaries = len(generated_summaries)
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
				mean_score[metric] = (values * 100 / num_summaries).tolist()
			scores.append(mean_score)
		return scores

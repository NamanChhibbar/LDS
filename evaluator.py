import os
import json
from warnings import filterwarnings

from nltk import sent_tokenize
from transformers import (
	BartTokenizer, BartForConditionalGeneration,
	T5Tokenizer, T5ForConditionalGeneration,
	GPT2TokenizerFast
)
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from utils.helpers import TextProcessor, TextSegmenter, get_device, count_words
from utils.encoders import (
	TruncateMiddle, UniformSampler, SentenceSampler, RemoveRedundancy
)
from utils.pipelines import SummarizationPipeline, OpenAIPipeline
from utils.evaluator_utils import Evaluator

filterwarnings("ignore")


def main() -> None:
	load_dotenv()

	data_dir = "/home/nchibbar/Data"
	# data_dir = "/Users/naman/Workspace/Data/Long-Document-Summarization"
	out_dir = f"{data_dir}/GovReport/processed"
	crs_files = os.listdir(f"{data_dir}/GovReport/crs")
	results_path = f"{data_dir}/govreport-results.json"

	# Sentence transformer
	sent_dir = f"{data_dir}/Models/Sent-Transformer"
	sent_encoder = SentenceTransformer(sent_dir)

	# BART
	bart_dir = f"{data_dir}/Models/BART"
	bart_fine_tuned = f"{data_dir}/Models/BART-GovReport-SentenceSampler"
	bart_tokenizer = BartTokenizer.from_pretrained(bart_dir)
	bart_model = BartForConditionalGeneration.from_pretrained(bart_fine_tuned)
	bart_context_size = bart_model.config.max_position_embeddings

	# T5
	t5_dir = f"{data_dir}/Models/T5"
	t5_tokenizer = T5Tokenizer.from_pretrained(t5_dir)
	t5_model = T5ForConditionalGeneration.from_pretrained(t5_dir)
	t5_context_size = t5_model.config.n_positions

	# GPT 3.5 turbo tokenizer
	gpt_dir = f"{data_dir}/Models/GPT-3.5-turbo-tokenizer"
	gpt_tokenizer = GPT2TokenizerFast.from_pretrained(gpt_dir)
	gpt_model = "gpt-3.5-turbo"
	gpt_context_size = 4096

	preprocessor = TextProcessor(preprocessing=True)
	postprocessor = None

	min_words = 50_000
	texts, summaries = [], []
	for file in crs_files:
		with open(f"{out_dir}/{file}") as fp:
			data = json.load(fp)
		if count_words(data["text"]) > min_words:
			texts.append(data["text"])
			summaries.append(data["summary"])

	segment_min_words = 20
	sent_segmenter = TextSegmenter(sent_tokenize, segment_min_words)

	head_size = .5
	threshold = .7
	seed = 69
	device = get_device()
	# device = "cpu"
	system_prompt = "You will be given some segments of a very long document. Your task is to summarize the entire document as a whole by extracting key information and ideas from the segments. Generate a detailed, concise, and coherent summary in 500 words. Do not refer to the document in the summary in any way."

	bart_encoders = [
		TruncateMiddle(
			bart_tokenizer, bart_context_size, head_size, preprocessor, True
		),
		UniformSampler(
			bart_tokenizer, bart_context_size, sent_segmenter, preprocessor,
			True, seed
		),
		SentenceSampler(
			bart_tokenizer, bart_context_size, sent_segmenter, sent_encoder,
			preprocessor, True, threshold, device, seed
		),
		RemoveRedundancy(
			bart_tokenizer, bart_context_size, sent_segmenter, sent_encoder,
			preprocessor, True, threshold, device, seed
		)
	]
	t5_encoders = [
		TruncateMiddle(
			t5_tokenizer, t5_context_size, head_size, preprocessor, True
		),
		UniformSampler(
			t5_tokenizer, t5_context_size, sent_segmenter, preprocessor,
			True, seed
		),
		SentenceSampler(
			t5_tokenizer, t5_context_size, sent_segmenter, sent_encoder,
			preprocessor, True, device=device, seed=seed
		),
		RemoveRedundancy(
			t5_tokenizer, t5_context_size, sent_segmenter, sent_encoder,
			preprocessor, True, device=device, seed=seed
		)
	]
	gpt_encoders = [
		TruncateMiddle(
			gpt_tokenizer, gpt_context_size, head_size, preprocessor, True
		),
		UniformSampler(
			gpt_tokenizer, gpt_context_size, sent_segmenter, preprocessor,
			True, seed
		),
		SentenceSampler(
			gpt_tokenizer, gpt_context_size, sent_segmenter, sent_encoder,
			preprocessor, True, device=device, seed=seed
		),
		RemoveRedundancy(
			gpt_tokenizer, gpt_context_size, sent_segmenter, sent_encoder,
			preprocessor, True, device=device, seed=seed
		)
	]
	bart_pipelines = [
		SummarizationPipeline(
			bart_model, enc, bart_context_size, postprocessor, device
		) for enc in bart_encoders
	]
	t5_pipelines = [
		SummarizationPipeline(
			t5_model, enc, t5_context_size, postprocessor, device
		) for enc in t5_encoders
	]
	gpt_pipelines = [
		# OpenAIPipeline(
		# 	gpt_model, enc, system_prompt=system_prompt
		# ) for enc in gpt_encoders
	]
	pipelines = bart_pipelines + t5_pipelines + gpt_pipelines

	batch_size = 3
	num_workers = min(len(pipelines), os.cpu_count())
	num_workers = 0

	evaluator = Evaluator(pipelines, num_workers, device)
	results = evaluator(texts, summaries, batch_size)

	with open(results_path, "w") as fp:
		json.dump(results, fp, indent=2)



if __name__ == "__main__":
	main()
	exit(0)

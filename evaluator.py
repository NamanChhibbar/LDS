import os
import json
from warnings import filterwarnings

from nltk import sent_tokenize
from transformers import (
	BartTokenizer, BartForConditionalGeneration,
	T5Tokenizer, T5ForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer

from utils.helpers import TextProcessor, TextSegmenter, get_device, count_words
from utils.encoders import (
	TruncateMiddle, UniformSampler, SegmentSampler, RemoveRedundancy
)
from utils.pipelines import SummarizationPipeline
from utils.evaluator_utils import Evaluator



def main() -> None:

	filterwarnings("ignore")

	data_dir = "/Users/naman/Workspace/Data/Long-Document-Summarization"
	data_dir = "/home/nchibbar/Data"
	out_dir = f"{data_dir}/GovReport/processed"
	crs_files = os.listdir(f"{data_dir}/GovReport/crs")
	results_path = f"{data_dir}/govreport-results2.json"

	# Sentence transformer
	# Automatically loads into gpu if available
	sent_dir = f"{data_dir}/Models/Sent-Transformer"
	sent_encoder = SentenceTransformer(sent_dir)

	# BART
	bart_dir = f"{data_dir}/Models/BART"
	bart_tokenizer = BartTokenizer.from_pretrained(bart_dir)
	bart_model = BartForConditionalGeneration.from_pretrained(bart_dir)
	bart_context_size = bart_model.config.max_position_embeddings

	# T5
	t5_dir = f"{data_dir}/Models/T5"
	t5_tokenizer = T5Tokenizer.from_pretrained(t5_dir)
	t5_model = T5ForConditionalGeneration.from_pretrained(t5_dir)
	t5_context_size = t5_model.config.n_positions

	preprocessor = TextProcessor(preprocessing=True)
	postprocessor = None

	min_words = 4_000
	max_words = 20_000
	max_texts = 100
	texts, summaries = [], []
	num_texts = 0
	for file in crs_files:
		with open(f"{out_dir}/{file}") as fp:
			data = json.load(fp)
		if min_words < count_words(data["text"]) < max_words:
			texts.append(data["text"])
			summaries.append(data["summary"])
			num_texts += 1
		if num_texts == max_texts:
			break

	print(f"Number of texts: {len(texts)}")

	segment_min_words = 20
	sent_segmenter = TextSegmenter(sent_tokenize, segment_min_words)

	min_token_frac = .5
	head_size = .5
	threshold = .8
	boost = .02
	seed = 69
	device = get_device()
	# device = "cpu"

	bart_min_tokens = int(min_token_frac * bart_context_size)
	t5_min_tokens = int(min_token_frac * t5_context_size)

	bart_encoders = [
		TruncateMiddle(
			bart_tokenizer, bart_context_size, head_size, preprocessor, True
		),
		UniformSampler(
			bart_tokenizer, bart_min_tokens, bart_context_size, sent_segmenter,
			preprocessor, True, seed
		),
		SegmentSampler(
			bart_tokenizer, bart_min_tokens, bart_context_size, sent_segmenter,
			sent_encoder, preprocessor, True, threshold, boost, seed
		),
		RemoveRedundancy(
			bart_tokenizer, bart_min_tokens, bart_context_size, sent_segmenter,
			sent_encoder, preprocessor, True, threshold, seed
		)
	]
	t5_encoders = [
		TruncateMiddle(
			t5_tokenizer, t5_context_size, head_size, preprocessor, True
		),
		UniformSampler(
			t5_tokenizer, t5_min_tokens, t5_context_size, sent_segmenter,
			preprocessor, True, seed
		),
		SegmentSampler(
			t5_tokenizer, t5_min_tokens, t5_context_size, sent_segmenter,
			sent_encoder, preprocessor, True, threshold, boost, seed
		),
		RemoveRedundancy(
			t5_tokenizer, t5_min_tokens, t5_context_size, sent_segmenter,
			sent_encoder, preprocessor, True, threshold, seed
		)
	]
	min_summary_tokens = 400
	bart_pipelines = [
		SummarizationPipeline(
			bart_model, enc, min_summary_tokens, bart_context_size,
			postprocessor, device
		) for enc in bart_encoders
	]
	t5_pipelines = [
		SummarizationPipeline(
			t5_model, enc, min_summary_tokens, t5_context_size,
			postprocessor, device
		) for enc in t5_encoders
	]
	pipelines = bart_pipelines + t5_pipelines

	batch_size = 3

	evaluator = Evaluator(pipelines, device)
	results = evaluator(texts, summaries, batch_size)

	with open(results_path, "w") as fp:
		json.dump(results, fp, indent=2)



if __name__ == "__main__":
	main()
	exit(0)

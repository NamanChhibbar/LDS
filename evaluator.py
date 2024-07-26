import os
import json
from warnings import filterwarnings
from argparse import ArgumentParser, Namespace

from nltk import sent_tokenize
from transformers import (
	BartTokenizer, BartForConditionalGeneration,
	T5Tokenizer, T5ForConditionalGeneration,
	PegasusTokenizerFast, PegasusForConditionalGeneration
)
from sentence_transformers import SentenceTransformer

from utils.helpers import (
	TextProcessor, TextSegmenter,
	get_device, count_words, get_stop_words,
	STOP_WORDS
)
from utils.encoders import *
from utils.pipelines import SummarizationPipeline
from utils.evaluator_utils import Evaluator



def main() -> None:

	filterwarnings("ignore")
	args = get_arguments()

	model_name = args.model.lower()
	dataset_name = args.dataset.lower()
	min_words = args.min_words
	max_words = args.max_words
	max_texts = args.max_texts
	device = "cpu" if args.no_gpu else get_device()

	data_dir = "/Users/naman/Workspace/Data/Long-Document-Summarization"
	data_dir = "/home/nchibbar/Data"
	govreport_dir = f"{data_dir}/GovReport/processed"
	bigpatent_dir = f"{data_dir}/BigPatent/processed"
	results_path = f"{data_dir}/{model_name}-{dataset_name}.json"
	sent_dir = f"{data_dir}/Models/Sent-Transformer"
	bart_dir = f"{data_dir}/Models/BART"
	t5_dir = f"{data_dir}/Models/T5"
	pegasus_dir = f"{data_dir}/Models/PEGASUS"

	segment_min_words = 20
	min_token_frac = .5
	head_size = .5
	threshold = .8
	boost = .02
	num_keywords = 20
	seed = 69
	min_summary_tokens = 100

	batch_size = 5
	temperature = 2.
	repetition_penalty = 3.
	top_p = .95

	print("Loading text processors and segmenter...")
	preprocessor = TextProcessor(preprocessing=True)
	keywords_preprocessor = TextProcessor(
		only_words_nums = True,
		remove_nums = True
	)
	postprocessor = None
	text_segmenter = TextSegmenter(sent_tokenize, segment_min_words)

	print("Loading sentence encoder...")
	sent_encoder = SentenceTransformer(sent_dir, device=device)

	print("Loading tokenizer and model...")
	match model_name:

		case "bart":
			tokenizer = BartTokenizer.from_pretrained(bart_dir)
			model = BartForConditionalGeneration.from_pretrained(bart_dir)
			context_size = model.config.max_position_embeddings

		case "t5":
			tokenizer = T5Tokenizer.from_pretrained(t5_dir)
			model = T5ForConditionalGeneration.from_pretrained(t5_dir)
			context_size = model.config.n_positions

		case "pegasus":
			tokenizer = PegasusTokenizerFast.from_pretrained(pegasus_dir)
			model = PegasusForConditionalGeneration.from_pretrained(pegasus_dir)
			context_size = model.config.max_position_embeddings
		
		case _:
			raise ValueError(f"Invalid model name: {model_name}")
		
	print(f"Context size of model: {context_size}")

	print("Initializing encoders and pipelines...")
	stop_words = get_stop_words(extra_stop_words=STOP_WORDS)
	min_tokens = int(min_token_frac * context_size)

	encoders = [
		TruncateMiddle(
			tokenizer, context_size, 1, preprocessor
		),
		TruncateMiddle(
			tokenizer, context_size, head_size, preprocessor
		),
		UniformSampler(
			tokenizer, min_tokens, context_size, text_segmenter,
			preprocessor, seed
		),
		SegmentSampler(
			tokenizer, min_tokens, context_size, text_segmenter,
			sent_encoder, preprocessor, threshold, boost, seed
		),
		RemoveRedundancy(
			tokenizer, min_tokens, context_size, text_segmenter,
			sent_encoder, preprocessor, threshold, seed
		),
		RemoveRedundancy2(
			tokenizer, min_tokens, context_size, text_segmenter,
			sent_encoder, preprocessor, .6, seed
		),
		KeywordScorer(
			tokenizer, context_size, text_segmenter, sent_encoder,
			preprocessor, num_keywords, keywords_preprocessor,
			stop_words
		)
	]

	pipelines = [
		SummarizationPipeline(
			model, enc, postprocessor, min_summary_tokens,
			context_size, device, temperature, repetition_penalty, top_p
		) for enc in encoders
	]

	print("Loading data...")
	texts, summaries = [], []
	num_texts = 0
	match dataset_name:

		case "govreport":
			files = os.listdir(govreport_dir)
			for file in files:
				file_path = f"{govreport_dir}/{file}"
				with open(file_path) as fp:
					data = json.load(fp)
				if min_words < count_words(data["text"]) < max_words:
					texts.append(data["text"])
					summaries.append(data["summary"])
					num_texts += 1
				if num_texts == max_texts:
					break
			
		case "bigpatent":
			files = os.listdir(bigpatent_dir)
			for file in files:
				file_path = f"{bigpatent_dir}/{file}"
				with open(file_path) as fp:
					data = json.load(fp)
				for text, summary in zip(data["texts"], data["summaries"]):
					if min_words < count_words(text) < max_words:
						texts.append(text)
						summaries.append(summary)
						num_texts += 1
					if num_texts == max_texts:
						break
				if num_texts == max_texts:
					break
		
		case _:
			raise ValueError(f"Invalid dataset name: {dataset_name}")

	print(f"Using {num_texts} texts")

	print(f"Evaluating pipelines with device {device}...")
	evaluator = Evaluator(pipelines, device)
	results = evaluator(texts, summaries, batch_size)

	print(f"Saving results in {results_path}...")
	with open(results_path, "w") as fp:
		json.dump(results, fp, indent=2)


def get_arguments() -> Namespace:
	parser = ArgumentParser(description="Training script")

	# Command line arguments
	parser.add_argument(
		"--model", action="store", type=str, required=True,
		help="Model to use"
	)
	parser.add_argument(
		"--dataset", action="store", type=str, required=True,
		help="Dataset to use"
	)
	parser.add_argument(
		"--min-words", action="store", type=float, default=0,
		help="Minimum words in text"
	)
	parser.add_argument(
		"--max-words", action="store", type=float, default=float("inf"),
		help="Maximum words in text"
	)
	parser.add_argument(
		"--max-texts", action="store", type=float, default=float("inf"),
		help="Maximum texts to use"
	)
	parser.add_argument(
		"--no-gpu", action="store_true",
		help="Specify to NOT use GPU"
	)

	args = parser.parse_args()
	return args



if __name__ == "__main__":
	main()
	exit(0)

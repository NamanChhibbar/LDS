"""
Script to evaluate summarization pipelines on a dataset.

DO NOT IMPORT THIS SCRIPT DIRECTLY. IT IS INTENDED TO BE RUN AS A SCRIPT.
"""

import os
import json
import time
import warnings
import argparse as ap
import dotenv

import nltk
import transformers as tfm
import sentence_transformers as stfm

import configs as c
import encoders as e
import pipelines as p
import utils as u



def main() -> None:

	warnings.filterwarnings("ignore")
	args = get_arguments()

	model_name = args.model.lower()
	dataset_name = args.dataset.lower()
	batch_size = args.batch_size
	device = "cpu" if args.no_gpu else u.get_device(1000)
	time_only = args.time_only

	data_dir = f"{c.BASE_DIR}/{dataset_name}"
	sent_dir = f"{c.BASE_DIR}/Models/sent-transformer"
	model_dir = f"{c.MODELS_DIR}/{model_name}"
	results_path = f"{c.BASE_DIR}/{model_name}-{dataset_name}{"-times" if time_only else ""}.json"

	print("Loading text processors and segmenter...")
	preprocessor = u.TextProcessor(preprocessing=True)
	keywords_preprocessor = u.TextProcessor(
		only_words_nums = True,
		remove_nums = True
	)
	postprocessor = None
	text_segmenter = u.TextSegmenter(nltk.sent_tokenize, c.SEGMENT_MIN_WORDS)

	print("Loading sentence encoder...")
	sent_encoder = stfm.SentenceTransformer(sent_dir, device=device)

	print("Loading tokenizer and model...")
	match model_name:

		case "bart":
			tokenizer = tfm.BartTokenizer.from_pretrained(model_dir)
			model = tfm.BartForConditionalGeneration.from_pretrained(model_dir)
			context_size = model.config.max_position_embeddings

		case "t5":
			tokenizer = tfm.T5Tokenizer.from_pretrained(model_dir)
			model = tfm.T5ForConditionalGeneration.from_pretrained(model_dir)
			context_size = model.config.n_positions

		case "pegasus":
			tokenizer = tfm.PegasusTokenizerFast.from_pretrained(model_dir)
			model = tfm.PegasusForConditionalGeneration.from_pretrained(model_dir)
			context_size = model.config.max_position_embeddings

		case "gpt":
			if not dotenv.load_dotenv():
				raise FileNotFoundError(".env file not found")
			tokenizer = tfm.GPT2TokenizerFast.from_pretrained(model_dir)
			model = "gpt-3.5-turbo"
			context_size = 4096
		
		case _:
			raise ValueError(f"Invalid model name: {model_name}")
		
	print(f"Context size of model: {context_size}")

	print("Initializing encoders and pipelines...")
	stop_words = u.get_stop_words(c.EXTRA_STOP_WORDS)
	min_tokens = int(c.MIN_TOKEN_FRAC * context_size)

	encoders = [
		e.TruncateMiddle(
			tokenizer, context_size, 1, preprocessor
		),
		e.TruncateMiddle(
			tokenizer, context_size, c.HEAD_SIZE, preprocessor
		),
		e.UniformSampler(
			tokenizer, min_tokens, context_size, text_segmenter,
			preprocessor, c.SEED
		),
		e.SegmentSampler(
			tokenizer, min_tokens, context_size, text_segmenter,
			sent_encoder, preprocessor, c.THRESHOLD, c.PROB_BOOST, c.SEED
		),
		e.RemoveRedundancy(
			tokenizer, min_tokens, context_size, text_segmenter,
			sent_encoder, preprocessor, c.THRESHOLD, c.SEED
		),
		# RemoveRedundancy2(
		# 	tokenizer, min_tokens, context_size, text_segmenter,
		# 	sent_encoder, preprocessor, .4, seed
		# ),
		e.KeywordScorer(
			tokenizer, context_size, text_segmenter, sent_encoder,
			preprocessor, c.NUM_KEYWORDS, keywords_preprocessor,
			stop_words
		)
	]

	pipelines = [
		p.SummarizationPipeline(
			model, enc, postprocessor, c.MIN_SUMMARY_TOKENS,
			context_size, device, c.TEMPERATURE, c.REPETITION_PENALTY, c.TOP_P
		) for enc in encoders
	] if model_name != "gpt" else [
		p.OpenAIPipeline(
			model, enc, postprocessor, c.SYSTEM_PROMPT
		) for enc in encoders
	]

	texts, summaries = [], []
	num_texts = 0

	print("Loading data...")
	match dataset_name:

		case "govreport":
			files = os.listdir(data_dir)
			for file in files:
				file_path = f"{data_dir}/{file}"
				with open(file_path) as fp:
					data = json.load(fp)
				if c.MIN_WORDS < u.count_words(data["text"]) < c.MAX_WORDS:
					texts.append(data["text"])
					summaries.append(data["summary"])
					num_texts += 1
				if num_texts == c.MAX_TEXTS:
					break
			
		case "bigpatent":
			files = os.listdir(data_dir)
			for file in files:
				file_path = f"{data_dir}/{file}"
				with open(file_path) as fp:
					data = json.load(fp)
				for text, summary in zip(data["texts"], data["summaries"]):
					if c.MIN_WORDS < u.count_words(text) < c.MAX_WORDS:
						texts.append(text)
						summaries.append(summary)
						num_texts += 1
					if num_texts == c.MAX_TEXTS:
						break
				if num_texts == c.MAX_TEXTS:
					break
		
		case _:
			raise ValueError(f"Invalid dataset name: {dataset_name}")

	print(f"Using {num_texts} texts")

	results = {
		"min_words": c.MIN_WORDS,
		"max_words": c.MAX_WORDS,
		"max_texts": c.MAX_TEXTS
	}

	if time_only:
		all_times = []
		print("Timing encoders...")
		for i, encoder in enumerate(encoders):
			print(f"Timing encoder {i + 1}...")
			start = time.perf_counter()
			encoder(texts)
			time_taken = (time.perf_counter() - start) * 1000 / num_texts
			print(
				f"Encoder {i + 1} took {round(time_taken, c.FLT_PREC)} ms/text on average"
			)
			all_times.append(time)
		results["encoder_times"] = all_times

	else:
		print(f"Evaluating pipelines with device {device}...")
		evaluator = u.Evaluator(pipelines, device)
		evaluator_results = evaluator(texts, summaries, batch_size)
		results.update(evaluator_results)

	print(f"Saving results in {results_path}...")
	with open(results_path, "w") as fp:
		json.dump(results, fp, indent=2)



def get_arguments() -> ap.Namespace:

	parser = ap.ArgumentParser(description="Training script")

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
		"--batch-size", action="store", type=int, required=True,
		help="maximum size of a batch"
	)
	parser.add_argument(
		"--no-gpu", action="store_true",
		help="Specify to NOT use GPU"
	)
	parser.add_argument(
		"--time-only", action="store_true",
		help="Specify to ONLY time encoders"
	)

	args = parser.parse_args()
	return args



if __name__ == "__main__":
	main()
	exit(0)

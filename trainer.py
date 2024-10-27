"""
Script to train a model using an encoder on a dataset.

DO NOT IMPORT THIS SCRIPT DIRECTLY. IT IS INTENDED TO BE RUN AS A SCRIPT.
"""

import os
import json
from warnings import filterwarnings
from argparse import ArgumentParser, Namespace

import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer

from configs import (
	BASE_DIR, MODELS_DIR, GPU_USAGE_TOLERANCE, MIN_TOKEN_FRAC,
	MIN_WORDS, MAX_WORDS, MAX_TEXTS, HEAD_SIZE, SEGMENT_MIN_WORDS,
	SEED, LEARNING_RATE, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
	FLT_PREC, SPACES, THRESHOLD, PROB_BOOST, NUM_KEYWORDS, EXTRA_STOP_WORDS
)
from encoders import (
	TruncateMiddle, UniformSampler, SegmentSampler,
	RemoveRedundancy, KeywordScorer
)
from utils import (
	get_device, get_stop_words, count_words, train_model,
	TextProcessor, TextSegmenter, SummarizationDataset
)



def main() -> None:

	filterwarnings("ignore")

	# Get command line arguments
	# See function get_arguments for descriptions
	args = get_arguments()
	model_name = args.model.lower()
	dataset = args.dataset.lower()
	encoder_name = args.encoder.lower()
	shuffle = args.no_shuffle
	batch_size = args.batch_size
	epochs = args.epochs
	device = "cpu" if args.no_gpu else get_device(GPU_USAGE_TOLERANCE)

	# All paths that are needed to be hard coded
	data_dir = f"{BASE_DIR}/{dataset}"
	sent_dir = f"{MODELS_DIR}/sent-transformer"
	model_dir = f"{MODELS_DIR}/{model_name}"
	save_dir = f"{MODELS_DIR}/{model_name}-{dataset}-{encoder_name}"
	train_history_path = f"{BASE_DIR}/{model_name}-{dataset}-history.json"

	print("Loading tokenizer and model...")
	match model_name:

		case "bart" | "pegasus":
			tokenizer = AutoTokenizer.from_pretrained(model_dir)
			model = AutoModelForCausalLM.from_pretrained(model_dir)
			context_size = model.config.max_position_embeddings

		case "t5":
			tokenizer = AutoTokenizer.from_pretrained(model_dir)
			model = AutoModelForCausalLM.from_pretrained(model_dir)
			context_size = model.config.n_positions

		case _:
			raise ValueError(f"Invalid model name: {model_name}")
	
	min_tokens = int(MIN_TOKEN_FRAC * context_size)
	print(f"Context size of model: {context_size}")

	texts, summaries = [], []
	num_texts = 0

	print("Loading data...")
	match dataset:

		case "govreport":
			files = os.listdir(data_dir)
			for file in files:
				file_path = f"{data_dir}/{file}"
				with open(file_path) as fp:
					data = json.load(fp)
				if MIN_WORDS < count_words(data["text"]) < MAX_WORDS:
					texts.append(data["text"])
					summaries.append(data["summary"])
					num_texts += 1
				if num_texts == MAX_TEXTS:
					break

		case "bigpatent":
			files = os.listdir(data_dir)
			for file in files:
				file_path = f"{data_dir}/{file}"
				with open(file_path) as fp:
					data = json.load(fp)
				for text, summary in zip(data["texts"], data["summaries"]):
					if MIN_WORDS < count_words(text) < MAX_WORDS:
						texts.append(text)
						summaries.append(summary)
						num_texts += 1
					if num_texts == MAX_TEXTS:
						break
				if num_texts == MAX_TEXTS:
					break

		case _:
			raise ValueError(f"Invalid dataset name: {dataset}")
	
	print(f"Using {num_texts} texts")

	print("Initializing encoder...")
	preprocessor = TextProcessor(preprocessing=True)
	text_segmenter = TextSegmenter(nltk.sent_tokenize, SEGMENT_MIN_WORDS)
	keywords_preprocessor = TextProcessor(
		only_words_nums = True,
		remove_nums = True
	)
	stop_words = get_stop_words(EXTRA_STOP_WORDS)

	match encoder_name:

		case "truncatemiddle":
			encoder = TruncateMiddle(
				tokenizer, context_size, HEAD_SIZE, preprocessor
			)

		case "uniformsampler":
			encoder = UniformSampler(
				tokenizer, min_tokens, context_size, text_segmenter,
				preprocessor, SEED
			)

		case "segmentsampler":
			sent_encoder = SentenceTransformer(sent_dir, device=device)
			encoder = SegmentSampler(
				tokenizer, min_tokens, context_size, text_segmenter,
				sent_encoder, preprocessor, THRESHOLD, PROB_BOOST, SEED
			)

		case "removeredundancy":
			sent_encoder = SentenceTransformer(sent_dir, device=device)
			encoder = RemoveRedundancy(
				tokenizer, min_tokens, context_size, text_segmenter,
				sent_encoder, preprocessor, THRESHOLD, SEED
			)

		case "keywordscorer":
			sent_encoder = SentenceTransformer(sent_dir, device=device)
			encoder = KeywordScorer(
				tokenizer, context_size, text_segmenter, sent_encoder,
				preprocessor, NUM_KEYWORDS, keywords_preprocessor,
				stop_words
			)
		
		case _:
			raise ValueError(f"Invalid encoder name: {encoder_name}")

	print("Initializing dataset...")
	dataset = SummarizationDataset(
		texts, encoder, batch_size, summaries,
		context_size, shuffle, SEED
	)

	# Adam optimizer with weight decay
	optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

	# Reduces LR when a tracked metric stops improving
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="min",
		factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
	)

	print(f"Starting training with device {device}...\n")
	train_history, successful = train_model(
		model, dataset, epochs, optimizer, scheduler,
		device, FLT_PREC, SPACES
	)

	if not successful:
		input("Press enter to save model")

	print(f"\nSaving model in {save_dir}...")
	model.save_pretrained(save_dir)

	print(f"Saving training history in {train_history_path}...")
	dirs, _ = os.path.split(train_history_path)
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	with open(train_history_path, "w") as fp:
		json.dump({
			"train-history": train_history
		}, fp, indent=2)

	print("Finished")



def get_arguments() -> Namespace:
	parser = ArgumentParser(description="Training script")

	# Command line arguments
	parser.add_argument(
		"--model", action="store", type=str, required=True,
		help="model to train"
	)
	parser.add_argument(
		"--dataset", action="store", type=str, required=True,
		help="dataset to train on"
	)
	parser.add_argument(
		"--encoder", action="store", type=str, required=True,
		help="encoder to use"
	)
	parser.add_argument(
		"--batch-size", action="store", type=int, required=True,
		help="maximum size of a batch"
	)
	parser.add_argument(
		"--epochs", action="store", type=int, required=True,
		help="number of epochs to train for"
	)
	parser.add_argument(
		"--no-gpu", action="store_true",
		help="specify to NOT use GPU"
	)
	parser.add_argument(
		"--no-shuffle", action="store_false",
		help="specify to NOT shuffle data in the dataset"
	)
	parser.add_argument(
		"--seed", action="store", type=int,
		help="use a manual seed for output reproducibility"
	)

	args = parser.parse_args()
	return args



if __name__ == "__main__":
	main()
	exit(0)

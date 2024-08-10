import os
import json
import warnings
import argparse as ap

import nltk
import transformers as tfm
import torch
import sentence_transformers as stfm

import configs as c
import utils.helpers as h
import utils.encoders as e
import utils.trainer_utils as tu



def main() -> None:

	warnings.filterwarnings("ignore")

	# Get command line arguments
	# See function get_arguments for descriptions
	args = get_arguments()
	model_name = args.model.lower()
	dataset = args.dataset.lower()
	encoder_name = args.encoder.lower()
	shuffle = args.no_shuffle
	batch_size = args.batch_size
	epochs = args.epochs
	device = "cpu" if args.no_gpu else h.get_device(c.GPU_USAGE_TOLERANCE)

	# All paths that are needed to be hard coded
	data_dir = f"{c.BASE_DIR}/{dataset}"
	sent_dir = f"{c.MODELS_DIR}/sent-transformer"
	model_dir = f"{c.MODELS_DIR}/{model_name}"
	save_dir = f"{c.MODELS_DIR}/{model_name}-{dataset}-{encoder_name}"
	train_history_path = f"{c.BASE_DIR}/{model_name}-{dataset}-history.json"

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

		case _:
			raise ValueError(f"Invalid model name: {model_name}")
	
	min_tokens = int(c.MIN_TOKEN_FRAC * context_size)
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
				if c.MIN_WORDS < h.count_words(data["text"]) < c.MAX_WORDS:
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
					if c.MIN_WORDS < h.count_words(text) < c.MAX_WORDS:
						texts.append(text)
						summaries.append(summary)
						num_texts += 1
					if num_texts == c.MAX_TEXTS:
						break
				if num_texts == c.MAX_TEXTS:
					break

		case _:
			raise ValueError(f"Invalid dataset name: {dataset}")
	
	print(f"Using {num_texts} texts")

	print("Initializing encoder...")
	preprocessor = h.TextProcessor(preprocessing=True)
	text_segmenter = h.TextSegmenter(nltk.sent_tokenize, c.SEGMENT_MIN_WORDS)
	keywords_preprocessor = h.TextProcessor(
		only_words_nums = True,
		remove_nums = True
	)
	stop_words = h.get_stop_words(c.EXTRA_STOP_WORDS)

	match encoder_name:

		case "truncatemiddle":
			encoder = e.TruncateMiddle(
				tokenizer, context_size, c.HEAD_SIZE, preprocessor
			)

		case "uniformsampler":
			encoder = e.UniformSampler(
				tokenizer, min_tokens, context_size, text_segmenter,
				preprocessor, c.SEED
			)

		case "segmentsampler":
			sent_encoder = stfm.SentenceTransformer(sent_dir, device=device)
			encoder = e.SegmentSampler(
				tokenizer, min_tokens, context_size, text_segmenter,
				sent_encoder, preprocessor, c.THRESHOLD, c.PROB_BOOST, c.SEED
			)

		case "removeredundancy":
			sent_encoder = stfm.SentenceTransformer(sent_dir, device=device)
			encoder = e.RemoveRedundancy(
				tokenizer, min_tokens, context_size, text_segmenter,
				sent_encoder, preprocessor, c.THRESHOLD, c.SEED
			)

		case "keywordscorer":
			sent_encoder = stfm.SentenceTransformer(sent_dir, device=device)
			encoder = e.KeywordScorer(
				tokenizer, context_size, text_segmenter, sent_encoder,
				preprocessor, c.NUM_KEYWORDS, keywords_preprocessor,
				stop_words
			)
		
		case _:
			raise ValueError(f"Invalid encoder name: {encoder_name}")

	print("Initializing dataset...")
	dataset = tu.SummarizationDataset(
		texts, encoder, batch_size, summaries,
		context_size, shuffle, c.SEED
	)

	# Adam optimizer with weight decay
	optimizer = torch.optim.AdamW(model.parameters(), c.LEARNING_RATE)

	# Reduces LR when a tracked metric stops improving
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="min",
		factor=c.SCHEDULER_FACTOR, patience=c.SCHEDULER_PATIENCE
	)

	print(f"Starting training with device {device}...\n")
	train_history, successful = tu.train_model(
		model, dataset, epochs, optimizer, scheduler,
		device, c.FLT_PREC, c.SPACES
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



def get_arguments() -> ap.Namespace:
	parser = ap.ArgumentParser(description="Training script")

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

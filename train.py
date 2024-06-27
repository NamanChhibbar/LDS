import os
import json
import pickle
from argparse import ArgumentParser, Namespace

from nltk import sent_tokenize
from transformers import (
	BartTokenizer, BartForConditionalGeneration,
	# T5Tokenizer, T5ForConditionalGeneration
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sentence_transformers import SentenceTransformer

from utils.helpers import (
	SummarizationDataset, TextProcessor, train_model,
	get_device, count_words
)
from utils.pipelines import SentenceSampler




def main() -> None:
	# Get command line arguments
	args = get_arguments()

	# All paths that are needed to be hard coded
	data_dir = "/home/nchibbar/Data"
	crs_dir = f"{data_dir}/GovReport/crs-processed"
	sent_dir = f"{data_dir}/Models/Sent-Transformer"
	bart_dir = f"{data_dir}/Models/BART"
	save_dir = f"{data_dir}/Models/BART-GovReport-SentenceSampler"
	# t5_dir = f"{data_dir}/Models/T5"
	# save_dir = f"{data_dir}/Models/T5-GovReport-SentenceSampler"
	train_history_path = f"{data_dir}/train-history/bart-history.pkl"

	# Use the command line arguments
	# See function get_arguments for description
	max_words = float("inf") if args.max_words is None else args.max_words
	shuffle = args.no_shuffle
	batch_size = args.batch_size
	use_cache = args.no_use_cache
	threshold = .7 if args.threshold is None else args.threshold
	lr = 1e-3 if args.learning_rate is None else args.learning_rate
	factor = .1 if args.factor is None else args.factor
	patience = 5 if args.patience is None else args.patience
	epochs = args.epochs
	device = get_device() if args.use_gpu else "cpu"
	seed = args.seed
	flt_prec = 4 if args.float_precision is None else args.float_precision

	print("Loading tokenizer and model...")
	# BART
	tokenizer = BartTokenizer.from_pretrained(bart_dir)
	model = BartForConditionalGeneration.from_pretrained(bart_dir)
	context_size = model.config.max_position_embeddings

	# T5
	# tokenizer = T5Tokenizer.from_pretrained(t5_dir)
	# model = T5ForConditionalGeneration.from_pretrained(t5_dir)
	# context_size = model.config.n_positions

	print("Loading data...")
	crs_files = os.listdir(crs_dir)
	texts, summaries = [], []
	for file in crs_files:
		with open(f"{crs_dir}/{file}") as fp:
			data = json.load(fp)
		if count_words(data["text"]) < max_words:
			texts.append(data["text"])
			summaries.append(data["summary"])

	print("Creating dataset...")
	preprocessor = TextProcessor(preprocessing=True)
	sent_encoder = SentenceTransformer(sent_dir)
	encoder = SentenceSampler(
		tokenizer, context_size, sent_tokenize, sent_encoder,
		preprocessor, threshold, device, seed
	)
	dataset = SummarizationDataset(
		texts, encoder, batch_size, summaries,
		context_size, use_cache, shuffle, seed
	)

	# Adam optimizer with weight decay
	optimizer = AdamW(model.parameters(), lr)
	# Reduces LR when a tracked metric stops improving
	scheduler = ReduceLROnPlateau(
		optimizer, mode="min", factor=factor, patience=patience
	)

	print(f"Using device {device}")
	print("Starting training...\n")
	train_history = train_model(
		model, dataset, epochs, optimizer, scheduler, device, flt_prec
	)
	print("\nSaving model...")
	model.save_pretrained(save_dir)
	print(f"Saving training history in {train_history_path}...")
	with open(train_history_path, "wb") as fp:
		pickle.dump(train_history, fp)




def get_arguments() -> Namespace:
	parser = ArgumentParser(description="Training script")

	parser.add_argument(
		"--batch-size", action="store", type=int, required=True,
		help="Maximum size of a batch"
	)
	parser.add_argument(
		"--epochs", action="store", type=int, required=True,
		help="Number of epochs to train for"
	)
	parser.add_argument(
		"--max-words", action="store", type=int,
		help="Maximum words allowed in text"
	)
	parser.add_argument(
		"--no-shuffle", action="store_false",
		help="Specify to not shuffle data in the dataset"
	)
	parser.add_argument(
		"--no-use-cache", action="store_false",
		help="Specify to not use cache to store processed inputs"
	)
	parser.add_argument(
		"--threshold", action="store", type=float,
		help="Maximum similarity threshold to pick sentences in "
		"sentence sampling pipelines"
	)
	parser.add_argument(
		"--learning-rate", action="store", type=float,
		help="Initial learning rate in optimizer"
	)
	parser.add_argument(
		"--factor", action="store", type=float,
		help="Factor parameter for ReduceLROnPlateau scheduler"
	)
	parser.add_argument(
		"--patience", action="store", type=int,
		help="Patience parameter for ReduceLROnPlateau scheduler"
	)
	parser.add_argument(
		"--use-gpu", action="store_true",
		help="Specify to use GPU, if available"
	)
	parser.add_argument(
		"--seed", action="store", type=int,
		help="Use a manual seed for output reproducibility"
	)
	parser.add_argument(
		"--float-precision", action="store", type=int,
		help="Number of decimal places to show in floating points"
	)
	return parser.parse_args()



if __name__ == "__main__":
	main()
	exit(0)

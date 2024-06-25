from argparse import ArgumentParser, Namespace
import os
import json
import pickle
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
	args = get_arguments()

	data_dir = "/home/nchibbar/Data"
	crs_dir = f"{data_dir}/GovReport/crs-processed"
	crs_files = os.listdir(crs_dir)
	sent_dir = f"{data_dir}/Models/Sent-Transformer"
	bart_dir = f"{data_dir}/Models/BART"
	save_dir = f"{data_dir}/Models/BART-GovReport-SentenceSampler"
	# t5_dir = f"{data_dir}/Models/T5"
	# save_dir = f"{data_dir}/Models/T5-GovReport-SentenceSampler"
	train_history_path = f"{data_dir}/train-history/bart-history.pkl"

	max_words = float("inf") if args.max_words is None else args.max_words
	shuffle = args.no_shuffle
	batch_size = args.batch_size
	use_cache = args.no_cache
	threshold = .7 if args.threshold is None else args.threshold
	lr = 1e-3 if args.learning_rate is None else args.learning_rate
	factor = .1 if args.factor is None else args.factor
	patience = 5 if args.patience is None else args.patience
	epochs = args.epochs
	device = get_device() if args.use_gpu else "cpu"
	seed = args.seed
	flt_prec = 4 if args.float_precision is None else args.float_precision

	print("Loading tokenizer and model...")
	tokenizer = BartTokenizer.from_pretrained(bart_dir)
	model = BartForConditionalGeneration.from_pretrained(bart_dir)
	context_size = model.config.max_position_embeddings
	# tokenizer = T5Tokenizer.from_pretrained(t5_dir)
	# model = T5ForConditionalGeneration.from_pretrained(t5_dir)
	# context_size = model.config.n_positions
	bos_id = tokenizer.bos_token_id
	eos_id = tokenizer.eos_token_id

	print("Loading data...")
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
		bos_id, eos_id, preprocessor, threshold, device, seed
	)
	dataset = SummarizationDataset(
		texts, encoder, batch_size, summaries,
		context_size, use_cache, shuffle, seed
	)

	optimizer = AdamW(model.parameters(), lr)
	scheduler = ReduceLROnPlateau(
		optimizer, mode="min", factor=factor, patience=patience
	)

	print(f"Using device {device}")
	print("Starting training...\n")
	loss_history = train_model(
		model, dataset, epochs, optimizer, scheduler, device, flt_prec
	)
	print("\nSaving model...")
	model.save_pretrained(save_dir)
	print(f"Saving training history in {train_history_path}...")
	with open(train_history_path, "wb") as fp:
		pickle.dump(loss_history, fp)

def get_arguments() -> Namespace:
	parser = ArgumentParser(description="Training script")

	parser.add_argument(
		"--batch-size", action="store", type=int, required=True,
		help="Maximum size of a batch"
	)
	parser.add_argument(
		"--epochs", action="store", type=int, required=True,
		help="Number of epochs"
	)
	parser.add_argument(
		"--max-words", action="store", type=int,
		help="Maximum words in text"
	)
	parser.add_argument(
		"--no-shuffle", action="store_false",
		help="Specify to not shuffle data"
	)
	parser.add_argument(
		"--no-cache", action="store_false",
		help="Specify to not use cache"
	)
	parser.add_argument(
		"--threshold", action="store", type=float,
		help="Threshold for encoder"
	)
	parser.add_argument(
		"--learning-rate", action="store", type=float,
		help="Learning rate for optimizer"
	)
	parser.add_argument(
		"--factor", action="store", type=float,
		help="Factor for ReduceLROnPlateau scheduler"
	)
	parser.add_argument(
		"--patience", action="store", type=int,
		help="Patience for ReduceLROnPlateau scheduler"
	)
	parser.add_argument(
		"--use-gpu", action="store_true",
		help="Specify to use GPU, if available"
	)
	parser.add_argument(
		"--seed", action="store", type=int,
		help="Seed to use"
	)
	parser.add_argument(
		"--float-precision", action="store", type=int,
		help="Number of decimal places to show in floating points"
	)
	return parser.parse_args()


if __name__ == "__main__":
	main()
	exit(0)

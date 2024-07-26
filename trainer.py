import os
import json
from argparse import ArgumentParser, Namespace

from nltk import sent_tokenize
from transformers import (
	BartTokenizer, BartForConditionalGeneration,
	T5Tokenizer, T5ForConditionalGeneration
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sentence_transformers import SentenceTransformer

from utils.helpers import TextProcessor, get_device, count_words
from utils.encoders import SegmentSampler
from utils.trainer_utils import SummarizationDataset, train_model



def main() -> None:

	# Get command line arguments
	# See function get_arguments for descriptions
	args = get_arguments()
	model_name = args.model.lower()
	dataset = args.dataset.lower()
	min_words = args.min_words
	max_words = args.max_words
	max_texts = args.max_texts
	shuffle = args.no_shuffle
	batch_size = args.batch_size
	use_cache = args.no_cache
	threshold = args.threshold
	lr = args.learning_rate
	factor = args.factor
	patience = args.patience
	epochs = args.epochs
	device = "cpu" if args.no_gpu else get_device()
	seed = args.seed
	flt_prec = args.float_precision

	# All paths that are needed to be hard coded
	data_dir = "/home/nchibbar/Data"
	govreport_dir = f"{data_dir}/GovReport/processed"
	bigpatent_dir = f"{data_dir}/BigPatent/processed"
	sent_dir = f"{data_dir}/Models/Sent-Transformer"
	bart_dir = f"{data_dir}/Models/BART"
	t5_dir = f"{data_dir}/Models/T5"
	save_dir = f"{data_dir}/Models/{model_name}-{dataset}-SegmentSampler"
	train_history_path = f"{data_dir}/{model_name}-{dataset}-history.json"

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

		case _:
			raise ValueError(f"Invalid model name: {model_name}")

	print("Loading data...")
	texts, summaries = [], []
	num_texts = 0
	match dataset:

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
			raise ValueError(f"Invalid dataset name: {dataset}")
	
	print(f"Using {num_texts} texts")

	print("Creating dataset...")
	preprocessor = TextProcessor(preprocessing=True)
	sent_encoder = SentenceTransformer(sent_dir)
	encoder = SegmentSampler(
		tokenizer=tokenizer, max_tokens=context_size,
		sent_tokenizer=sent_tokenize, sent_encoder=sent_encoder,
		preprocessor=preprocessor, threshold=threshold, seed=seed
	)
	dataset = SummarizationDataset(
		texts, encoder, batch_size, summaries,
		context_size, use_cache, shuffle, seed
	)

	# Adam optimizer with weight decay
	optimizer = AdamW(model.parameters(), lr)

	# Reduces LR when a tracked metric stops improving
	scheduler = ReduceLROnPlateau(
		optimizer, mode="min",
		factor=factor, patience=patience
	)

	print(f"Starting training with device {device}...\n")
	train_history = train_model(
		model, dataset, epochs, optimizer, scheduler, device, flt_prec
	)

	print("\nSaving model...")
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
		help="Model to train"
	)
	parser.add_argument(
		"--dataset", action="store", type=str, required=True,
		help="Dataset to train on"
	)
	parser.add_argument(
		"--batch-size", action="store", type=int, required=True,
		help="Maximum size of a batch"
	)
	parser.add_argument(
		"--epochs", action="store", type=int, required=True,
		help="Number of epochs to train for"
	)
	parser.add_argument(
		"--no-gpu", action="store_true",
		help="Specify to NOT use GPU"
	)
	parser.add_argument(
		"--no-shuffle", action="store_false",
		help="Specify to NOT shuffle data in the dataset"
	)
	parser.add_argument(
		"--no-cache", action="store_false",
		help="Specify to NOT use cache to store processed inputs"
	)
	parser.add_argument(
		"--min-words", action="store", type=int, default=0,
		help="Minimum words allowed in text"
	)
	parser.add_argument(
		"--max-words", action="store", type=int, default=float("inf"),
		help="Maximum words allowed in text"
	)
	parser.add_argument(
		"--max-texts", action="store", type=int, default=float("inf"),
		help="Maximum texts to use"
	)
	parser.add_argument(
		"--threshold", action="store", type=float, default=.7,
		help="Maximum similarity threshold to pick sentences in "
		"SegmentSampler pipeline"
	)
	parser.add_argument(
		"--learning-rate", action="store", type=float, default=1e-3,
		help="Initial learning rate in optimizer"
	)
	parser.add_argument(
		"--factor", action="store", type=float, default=.1,
		help="Factor parameter for ReduceLROnPlateau scheduler"
	)
	parser.add_argument(
		"--patience", action="store", type=int, default=5,
		help="Patience parameter for ReduceLROnPlateau scheduler"
	)
	parser.add_argument(
		"--seed", action="store", type=int,
		help="Use a manual seed for output reproducibility"
	)
	parser.add_argument(
		"--float-precision", action="store", type=int, default=4,
		help="Number of decimal places to show in floating points"
	)

	args = parser.parse_args()
	return args



if __name__ == "__main__":
	main()
	exit(0)

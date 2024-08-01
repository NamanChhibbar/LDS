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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sentence_transformers import SentenceTransformer

from utils.helpers import (
	TextProcessor, TextSegmenter,
	get_device, count_words, get_stop_words,
	STOP_WORDS
)
from utils.encoders import *
from utils.trainer_utils import SummarizationDataset, train_model



def main() -> None:

	filterwarnings("ignore")

	# Get command line arguments
	# See function get_arguments for descriptions
	args = get_arguments()
	model_name = args.model.lower()
	dataset = args.dataset.lower()
	encoder_name = args.encoder.lower()
	min_words = args.min_words
	max_words = args.max_words
	max_texts = args.max_texts
	shuffle = args.no_shuffle
	batch_size = args.batch_size
	lr = args.learning_rate
	factor = args.factor
	patience = args.patience
	epochs = args.epochs
	device = "cpu" if args.no_gpu else get_device(1000)
	seed = args.seed
	flt_prec = args.float_precision

	# All paths that are needed to be hard coded
	data_dir = "/home/nchibbar/Data"
	govreport_dir = f"{data_dir}/GovReport/processed"
	bigpatent_dir = f"{data_dir}/BigPatent/processed"
	sent_dir = f"{data_dir}/Models/Sent-Transformer"
	bart_dir = f"{data_dir}/Models/BART"
	t5_dir = f"{data_dir}/Models/T5"
	pegasus_dir = f"{data_dir}/Models/PEGASUS"
	save_dir = f"{data_dir}/Models/{model_name}-{dataset}-SegmentSampler"
	train_history_path = f"{data_dir}/{model_name}-{dataset}-history.json"

	segment_min_words = 20
	min_token_frac = .5
	head_size = .5
	threshold = .8
	boost = .02
	num_keywords = 20

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
	
	min_tokens = int(min_token_frac * context_size)
	print(f"Context size of model: {context_size}")

	texts, summaries = [], []
	num_texts = 0

	print("Loading data...")
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

	print("Initializing encoder...")
	preprocessor = TextProcessor(preprocessing=True)
	text_segmenter = TextSegmenter(sent_tokenize, segment_min_words)
	keywords_preprocessor = TextProcessor(
		only_words_nums = True,
		remove_nums = True
	)
	stop_words = get_stop_words(extra_stop_words=STOP_WORDS)

	match encoder_name:

		case "truncatemiddle":
			encoder = TruncateMiddle(
				tokenizer, context_size, head_size, preprocessor
			)

		case "uniformsampler":
			encoder = UniformSampler(
				tokenizer, min_tokens, context_size, text_segmenter,
				preprocessor, seed
			)

		case "segmentsampler":
			sent_encoder = SentenceTransformer(sent_dir, device=device)
			encoder = SegmentSampler(
				tokenizer, min_tokens, context_size, text_segmenter,
				sent_encoder, preprocessor, threshold, boost, seed
			)

		case "removeredundancy":
			sent_encoder = SentenceTransformer(sent_dir, device=device)
			encoder = RemoveRedundancy(
				tokenizer, min_tokens, context_size, text_segmenter,
				sent_encoder, preprocessor, threshold, seed
			)

		case "keywordscorer":
			sent_encoder = SentenceTransformer(sent_dir, device=device)
			encoder = KeywordScorer(
				tokenizer, context_size, text_segmenter, sent_encoder,
				preprocessor, num_keywords, keywords_preprocessor,
				stop_words
			)
		
		case _:
			raise ValueError(f"Invalid encoder name: {encoder_name}")

	print("Initializing dataset...")
	dataset = SummarizationDataset(
		texts, encoder, batch_size, summaries,
		context_size, shuffle, seed
	)

	# Adam optimizer with weight decay
	optimizer = AdamW(model.parameters(), lr)

	# Reduces LR when a tracked metric stops improving
	scheduler = ReduceLROnPlateau(
		optimizer, mode="min",
		factor=factor, patience=patience
	)

	print(f"Starting training with device {device}...\n")
	train_history, successful = train_model(
		model, dataset, epochs, optimizer, scheduler, device, flt_prec
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
		"--min-words", action="store", type=int, default=0,
		help="minimum words allowed in text, 0 by default"
	)
	parser.add_argument(
		"--max-words", action="store", type=int, default=float("inf"),
		help="maximum words allowed in text, infinite by default"
	)
	parser.add_argument(
		"--max-texts", action="store", type=int, default=float("inf"),
		help="maximum texts to use, infinite by default"
	)
	parser.add_argument(
		"--learning-rate", action="store", type=float, default=1e-3,
		help="initial learning rate in optimizer, 1e-3 by default"
	)
	parser.add_argument(
		"--factor", action="store", type=float, default=.1,
		help="factor parameter for ReduceLROnPlateau scheduler, 0.1 by default"
	)
	parser.add_argument(
		"--patience", action="store", type=int, default=5,
		help="patience parameter for ReduceLROnPlateau scheduler, 5 by default"
	)
	parser.add_argument(
		"--seed", action="store", type=int,
		help="use a manual seed for output reproducibility"
	)
	parser.add_argument(
		"--float-precision", action="store", type=int, default=4,
		help="number of decimal places to show in floating points, 4 by default"
	)

	args = parser.parse_args()
	return args



if __name__ == "__main__":
	main()
	exit(0)

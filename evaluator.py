import os
import json
import pickle
from argparse import ArgumentParser, Namespace

from nltk import sent_tokenize
from transformers import (
	BartTokenizer, BartForConditionalGeneration,
	# T5Tokenizer, T5ForConditionalGeneration
)
from sentence_transformers import SentenceTransformer

from utils.helpers import TextProcessor, get_device, count_words
from utils.encoders import (
	TruncateMiddle, UniformSampler, SentenceSampler, RemoveRedundancy
)
from utils.pipelines import SummarizationPipeline
from utils.evaluator_utils import Evaluator



def main() -> None:
	# Get command line arguments
	args = get_arguments()

	# All paths that are needed to be hard coded
	data_dir = "/home/nchibbar/Data"
	crs_dir = f"{data_dir}/GovReport/crs-processed"
	sent_dir = f"{data_dir}/Models/Sent-Transformer"
	bart_dir = f"{data_dir}/Models/BART"
	# t5_dir = f"{data_dir}/Models/T5"
	# save_dir = f"{data_dir}/Models/T5-GovReport-SentenceSampler"
	scores_path = f"{data_dir}/scores/bart-scores.pkl"

	print("Loading tokenizer and model...")
	# BART
	tokenizer = BartTokenizer.from_pretrained(bart_dir)
	model = BartForConditionalGeneration.from_pretrained(bart_dir)
	context_size = model.config.max_position_embeddings

	# T5
	# tokenizer = T5Tokenizer.from_pretrained(t5_dir)
	# model = T5ForConditionalGeneration.from_pretrained(t5_dir)
	# context_size = model.config.n_positions

	# Use the command line arguments
	# See function get_arguments for descriptions
	device = get_device() if args.use_gpu else "cpu"
	max_words = float("inf") if args.max_words is None else args.max_words
	max_tokens = model.config.max_length if args.max_tokens is None \
		else args.max_tokens
	head_size = .25 if args.head_size is None else args.head_size
	threshold = .7 if args.threshold is None else args.threshold
	batch_size = args.batch_size
	num_workers = args.num_workers
	seed = args.seed

	print("Loading data...")
	crs_files = os.listdir(crs_dir)
	texts, summaries = [], []
	for file in crs_files:
		file = os.path.join(crs_dir, file)
		with open(file) as fp:
			data = json.load(fp)
		if count_words(data["text"]) < max_words:
			texts.append(data["text"])
			summaries.append(data["summary"])

	print(f"Using device {device}")
	print("Creating evaluator...")
	preprocessor = TextProcessor(preprocessing=True)
	postprocessor = None
	sent_encoder = SentenceTransformer(sent_dir)
	encoders = [
		TruncateMiddle(
			tokenizer=tokenizer, max_tokens=context_size,
			head_size=head_size, preprocessor=preprocessor
		),
		UniformSampler(
			tokenizer=tokenizer, max_tokens=context_size,
			sent_tokenizer=sent_tokenize, preprocessor=preprocessor,
			seed=seed
		),
		SentenceSampler(
			tokenizer=tokenizer, max_tokens=context_size,
			sent_tokenizer=sent_tokenize, sent_encoder=sent_encoder,
			preprocessor=preprocessor, threshold=threshold,
			device=device, seed=seed
		),
		RemoveRedundancy(
			tokenizer=tokenizer, max_tokens=context_size,
			sent_tokenizer=sent_tokenize, sent_encoder=sent_encoder,
			preprocessor=preprocessor, threshold=threshold,
			device=device, seed=seed
		)
	]
	pipelines = [
		SummarizationPipeline(
			model, encoder, max_tokens, postprocessor, device
		) for encoder in encoders
	]
	evaluator = Evaluator(
		pipelines, device=device
	)

	print("Generating summaries...")
	time_taken = evaluator.generate_summaries(texts, batch_size, num_workers)
	print("Getting scores...\n")
	bert_score = evaluator.get_bert_score(summaries)
	rouge_score = evaluator.get_rouge_score(summaries)
	scores = {
		"time-taken": time_taken,
		"bert-scores": bert_score,
		"rouge-scores": rouge_score
	}
	print(scores)
	print(f"Saving scores in {scores_path}...")
	dirs, _ = os.path.split(scores_path)
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	with open(scores_path, "wb") as fp:
		pickle.dump(scores, fp)



def get_arguments() -> Namespace:
	parser = ArgumentParser(description="Training script")

	parser.add_argument(
		"--use-gpu", action="store_true",
		help="Specify to use GPU, if available"
	)
	parser.add_argument(
		"--max-words", action="store", type=int,
		help="Maximum words allowed in text"
	)
	parser.add_argument(
		"--max-tokens", action="store", type=int,
		help="Maximum tokens allowed in summary"
	)
	parser.add_argument(
		"--head-size", action="store", type=float,
		help="Fraction of head tokens for TruncateMiddle pipeline"
	)
	parser.add_argument(
		"--threshold", action="store", type=float,
		help="Maximum similarity threshold to pick sentences in "
		"SentenceSampler pipeline"
	)
	parser.add_argument(
		"--batch-size", action="store", type=int,
		help="Maximum size of a batch"
	)
	parser.add_argument(
		"--num-workers", action="store", type=int,
		help="Number of subprocesses to generate summaries"
	)
	parser.add_argument(
		"--seed", action="store", type=int,
		help="Use a manual seed for output reproducibility"
	)
	args = parser.parse_args()
	return args



if __name__ == "__main__":
	main()
	exit(0)

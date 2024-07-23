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
	inf = float("inf")

	args = get_arguments()

	name = args.model.lower()

	data_dir = "/Users/naman/Workspace/Data/Long-Document-Summarization"
	data_dir = "/home/nchibbar/Data"
	govreport_dir = f"{data_dir}/GovReport/processed"
	govreport_files = os.listdir(govreport_dir)
	results_path = f"{data_dir}/govreport-{name}.json"
	sent_dir = f"{data_dir}/Models/Sent-Transformer"
	bart_dir = f"{data_dir}/Models/BART"
	t5_dir = f"{data_dir}/Models/T5"
	pegasus_dir = f"{data_dir}/Models/PEGASUS"

	# Sentence transformer
	# Automatically loads into gpu if available
	sent_encoder = SentenceTransformer(sent_dir)

	match name:

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
			raise ValueError(f"Invalid model name: {name}")

	# Preprocessors and postprocessor
	preprocessor = TextProcessor(preprocessing=True)
	keywords_preprocessor = TextProcessor(
		only_words_nums = True,
		remove_nums = True
	)
	postprocessor = None

	# Stop words
	stop_words = get_stop_words(extra_stop_words=STOP_WORDS)

	# Load data
	min_words = 20_000
	max_words = inf
	max_texts = inf

	texts, summaries = [], []
	num_texts = 0
	for file in govreport_files:
		file_path = f"{govreport_dir}/{file}"
		with open(file_path) as fp:
			data = json.load(fp)
		if min_words < count_words(data["text"]) < max_words:
			texts.append(data["text"])
			summaries.append(data["summary"])
			num_texts += 1
		if num_texts == max_texts:
			break

	print(f"Number of texts: {len(texts)}")

	segment_min_words = 20
	text_segmenter = TextSegmenter(sent_tokenize, segment_min_words)

	min_token_frac = .5
	head_size = .5
	threshold = .8
	boost = .02
	num_keywords = 20
	seed = 69
	device = get_device()
	# device = "cpu"
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
		KeywordScorer(
			tokenizer, context_size, text_segmenter, sent_encoder,
			preprocessor, num_keywords, keywords_preprocessor,
			stop_words
		)
	]
	min_summary_tokens = 300
	pipelines = [
		SummarizationPipeline(
			model, enc, postprocessor, min_summary_tokens,
			context_size, device
		) for enc in encoders
	]

	batch_size = 5
	evaluator = Evaluator(pipelines, device)
	results = evaluator(texts, summaries, batch_size)

	with open(results_path, "w") as fp:
		json.dump(results, fp, indent=2)


def get_arguments() -> Namespace:
	parser = ArgumentParser(description="Training script")

	parser.add_argument(
		"--model", action="store", type=str, required=True,
		help="Model to use"
	)
	args = parser.parse_args()
	return args



if __name__ == "__main__":
	main()
	exit(0)

'''
Script to evaluate summarization pipelines on a dataset.

DO NOT IMPORT THIS SCRIPT DIRECTLY. IT IS INTENDED TO BE RUN AS A SCRIPT.
'''

import os
import json
from time import perf_counter
from warnings import filterwarnings
import argparse as ap

import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from configs import (
	BASE_DIR, MODELS_DIR, OPENAI_API_KEY, MIN_WORDS, MAX_WORDS, MAX_TEXTS,
	SEGMENT_MIN_WORDS, MIN_TOKEN_FRAC, HEAD_SIZE, THRESHOLD, PROB_BOOST,
	NUM_KEYWORDS, SYSTEM_PROMPT, EXTRA_STOP_WORDS, MIN_SUMMARY_TOKENS,
	TEMPERATURE, REPETITION_PENALTY, TOP_P, GPU_USAGE_TOLERANCE, SEED, FLT_PREC
)
from encoders import (
	TruncateMiddle, UniformSampler, SegmentSampler,
	RemoveRedundancy, KeywordScorer
)
from pipelines import SummarizationPipeline, OpenAIPipeline
from utils import (
	get_device, get_stop_words, count_words,
	TextProcessor, TextSegmenter, Evaluator
)



def main() -> None:

	filterwarnings('ignore')
	args = get_arguments()

	model_name = args.model.lower()
	dataset_name = args.dataset.lower()
	batch_size = args.batch_size
	device = 'cpu' if args.no_gpu else get_device(GPU_USAGE_TOLERANCE)
	time_only = args.time_only

	data_dir = f'{BASE_DIR}/{dataset_name}'
	sent_dir = f'{BASE_DIR}/Models/sent-transformer'
	model_dir = f'{MODELS_DIR}/{model_name}'
	results_path = f'{BASE_DIR}/{model_name}-{dataset_name}{'-times' if time_only else ''}.json'

	print('Loading text processors and segmenter...')
	preprocessor = TextProcessor(preprocessing=True)
	keywords_preprocessor = TextProcessor(
		only_words_nums = True,
		remove_nums = True
	)
	postprocessor = None
	text_segmenter = TextSegmenter(nltk.sent_tokenize, SEGMENT_MIN_WORDS)

	print('Loading sentence encoder...')
	sent_encoder = SentenceTransformer(sent_dir, device=device)

	print('Loading tokenizer and model...')
	match model_name:

		case 'bart' | 'pegasus':
			tokenizer = AutoTokenizer.from_pretrained(model_dir)
			model = AutoModelForCausalLM.from_pretrained(model_dir)
			context_size = model.config.max_position_embeddings

		case 't5':
			tokenizer = AutoTokenizer.from_pretrained(model_dir)
			model = AutoModelForCausalLM.from_pretrained(model_dir)
			context_size = model.config.n_positions

		case 'gpt':
			if not OPENAI_API_KEY:
				raise RuntimeError('OpenAI API key not found')
			tokenizer = AutoTokenizer.from_pretrained(model_dir)
			model = 'gpt-3.5-turbo'
			context_size = 4096
		
		case _:
			raise ValueError(f'Invalid model name: {model_name}')
		
	print(f'Context size of model: {context_size}')

	print('Initializing encoders and pipelines...')
	stop_words = get_stop_words(EXTRA_STOP_WORDS)
	min_tokens = int(MIN_TOKEN_FRAC * context_size)

	encoders = [
		TruncateMiddle(
			tokenizer, context_size, 1, preprocessor
		),
		TruncateMiddle(
			tokenizer, context_size, HEAD_SIZE, preprocessor
		),
		UniformSampler(
			tokenizer, min_tokens, context_size, text_segmenter,
			preprocessor, SEED
		),
		SegmentSampler(
			tokenizer, min_tokens, context_size, text_segmenter,
			sent_encoder, preprocessor, THRESHOLD, PROB_BOOST, SEED
		),
		RemoveRedundancy(
			tokenizer, min_tokens, context_size, text_segmenter,
			sent_encoder, preprocessor, THRESHOLD, SEED
		),
		# RemoveRedundancy2(
		# 	tokenizer, min_tokens, context_size, text_segmenter,
		# 	sent_encoder, preprocessor, .4, seed
		# ),
		KeywordScorer(
			tokenizer, context_size, text_segmenter, sent_encoder,
			preprocessor, NUM_KEYWORDS, keywords_preprocessor,
			stop_words
		)
	]

	pipelines = [
		SummarizationPipeline(
			model, enc, postprocessor, MIN_SUMMARY_TOKENS,
			context_size, device, TEMPERATURE, REPETITION_PENALTY, TOP_P
		) for enc in encoders
	] if model_name != 'gpt' else [
		OpenAIPipeline(
			model, enc, OPENAI_API_KEY, postprocessor, SYSTEM_PROMPT
		) for enc in encoders
	]

	texts, summaries = [], []
	num_texts = 0

	print('Loading data...')
	match dataset_name:

		case 'govreport':
			files = os.listdir(data_dir)
			for file in files:
				file_path = f'{data_dir}/{file}'
				with open(file_path) as fp:
					data = json.load(fp)
				if MIN_WORDS < count_words(data['text']) < MAX_WORDS:
					texts.append(data['text'])
					summaries.append(data['summary'])
					num_texts += 1
				if num_texts == MAX_TEXTS:
					break
			
		case 'bigpatent':
			files = os.listdir(data_dir)
			for file in files:
				file_path = f'{data_dir}/{file}'
				with open(file_path) as fp:
					data = json.load(fp)
				for text, summary in zip(data['texts'], data['summaries']):
					if MIN_WORDS < count_words(text) < MAX_WORDS:
						texts.append(text)
						summaries.append(summary)
						num_texts += 1
					if num_texts == MAX_TEXTS:
						break
				if num_texts == MAX_TEXTS:
					break
		
		case _:
			raise ValueError(f'Invalid dataset name: {dataset_name}')

	print(f'Using {num_texts} texts')

	results = {
		'min_words': MIN_WORDS,
		'max_words': MAX_WORDS,
		'max_texts': MAX_TEXTS
	}

	if time_only:
		all_times = []
		print('Timing encoders...')
		for i, encoder in enumerate(encoders):
			print(f'Timing encoder {i + 1}...')
			start = perf_counter()
			encoder(texts)
			time_taken = (perf_counter() - start) * 1000 / num_texts
			print(
				f'Encoder {i + 1} took {round(time_taken, FLT_PREC)} ms/text on average'
			)
			all_times.append(time_taken)
		results['encoder_times'] = all_times

	else:
		print(f'Evaluating pipelines with device {device}...')
		evaluator = Evaluator(pipelines, device)
		evaluator_results = evaluator(texts, summaries, batch_size)
		results.update(evaluator_results)

	print(f'Saving results in {results_path}...')
	with open(results_path, 'w') as fp:
		json.dump(results, fp, indent=2)



def get_arguments() -> ap.Namespace:

	parser = ap.ArgumentParser(description='Training script')

	# Command line arguments
	parser.add_argument(
		'--model', action='store', type=str, required=True,
		help='Model to use'
	)
	parser.add_argument(
		'--dataset', action='store', type=str, required=True,
		help='Dataset to use'
	)
	parser.add_argument(
		'--batch-size', action='store', type=int, required=True,
		help='maximum size of a batch'
	)
	parser.add_argument(
		'--no-gpu', action='store_true',
		help='Specify to NOT use GPU'
	)
	parser.add_argument(
		'--time-only', action='store_true',
		help='Specify to ONLY time encoders'
	)

	args = parser.parse_args()
	return args



if __name__ == '__main__':
	main()
	exit(0)

'''
Contains helper functions.
'''

import subprocess

import numpy as np
import torch



def gpu_usage() -> list[int]:
	'''
	Get the current GPU memory usage.
	'''
	# Get output from nvidia-smi
	result = subprocess.check_output([
		'nvidia-smi',
		'--query-gpu=memory.used',
		'--format=csv,nounits,noheader'
	]).decode('utf-8').strip()

	# Extract memory used by GPUs in MiB
	gpu_memory = [int(mem) for mem in result.split('\n')]

	return gpu_memory


def get_device(threshold: int | float = 500) -> str:
	'''
	Returns a device with memory usage below `threshold`.
	'''
	if torch.cuda.is_available():
		usage = gpu_usage()
		cuda_ind = np.argmin(usage)
		return f'cuda:{cuda_ind}' if usage[cuda_ind] < threshold \
			else 'cpu'
	if torch.backends.mps.is_available():
		usage = torch.mps.driver_allocated_memory() / 1e6
		return 'mps' if usage < threshold else 'cpu'
	return 'cpu'


def count_words(text: str) -> int:
	words = text.split()
	num_words = len(words)
	return num_words


def count_tokens(
	texts: str | list[str],
	tokenizer
) -> tuple[int, list[int]] | tuple[int, list[list[int]]]:
	encodings = tokenizer(
		texts,
		add_special_tokens = False,
		verbose = False
	)['input_ids']
	num_tokens = len(encodings) if isinstance(texts, str) \
		else sum([len(encoding) for encoding in encodings])
	return num_tokens, encodings


def show_exception(exception: Exception) -> None:
	exc_class = exception.__class__.__name__
	exc_msg = str(exception)
	print(f'\nEncountered exception of type {exc_class}: {exc_msg}\n')


def clear_stdout(spaces: int = 100) -> None:
	print(f'\r{' ' * spaces}', end='\r')

import os
import json
import pickle
from nltk import sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.helpers import (
	SummarizationDataset, TextProcessor, train_model, get_device
)
from utils.pipelines import UniformSampler

def main() -> None:

	data_dir = "/home/nchibbar/Data"
	crs_dir = f"{data_dir}/GovReport/crs-processed"
	crs_files = os.listdir(crs_dir)
	t5_dir = f"{data_dir}/Models/T5"
	train_history_path = f"{data_dir}/train-history/t5-history.pkl"

	shuffle = True
	batch_size = 64
	lr = 1e-3
	factor = .1
	patience = 5
	epochs = 100
	seed = 69
	device = "cpu"
	# device = get_device()

	print("Loading data")
	texts_summaries = []
	for file in crs_files:
		with open(f"{crs_dir}/{file}") as fp:
			data = json.load(fp)
		texts_summaries.append((data["text"], data["summary"]))

	print("Loaded tokenizer and model")
	tokenizer = T5Tokenizer.from_pretrained(t5_dir)
	model = T5ForConditionalGeneration.from_pretrained(t5_dir)
	context_size = model.config.n_positions

	print("Processing dataset")
	preprocessor = TextProcessor(preprocessing=True)
	encoder = UniformSampler(
		tokenizer, context_size, sent_tokenize, preprocessor, seed
	)
	dataset = SummarizationDataset(
		texts_summaries, encoder, batch_size, shuffle
	)

	optimizer = AdamW(model.parameters(), lr)
	scheduler = ReduceLROnPlateau(
		optimizer, mode="min", factor=factor, patience=patience
	)

	print(f"Using device {device}")
	print("Starting training\n")
	loss_history = train_model(
		model, dataset, epochs, optimizer, scheduler, device
	)
	print("\nTraining completed")
	model.save_pretrained(t5_dir)
	print("Model saved")

	with open(train_history_path, "wb") as fp:
		pickle.dump(loss_history, fp)
	print(f"Training history saved in {train_history_path}")

if __name__ == "__main__":
	main()
	exit(0)

# Long Document Summarizer

This repository contains the official implementation of the work done in [this paper](https://arxiv.org/abs/2410.05903).

## Installing Dependencies

Create a python environment using the following commmand:

```sh
python -m venv .venv
```

This will create a virtual environment `.venv/` in your directory.

Activate the virtual environment by running:

```sh
source .venv/bin/activate
```

## Updating the Configurations

Update the default configurations in [configs.py](configs.py).
You will need to update the `OPENAI_API_KEY` in this file to use the OpenAIPipeline in [pipelines.py](pipelines.py).

## Using a Summarization Pipeline

There are two summarizaation pipelines available in [pipelines.py](pipelines.py).
To generate the summary for a given text, instantiate them with the necessary arguments and use the `generate_summaries` method.

## Using the Encoders independently

To use any of the encoding algorithm mentioned in the paper, import the necessary class from [encoders.py](encoders.py).
Refer to the documentation in the file for usage.

## Training a Model using an Encoder

Run [train.py](train.py) to train a model on an encoder.

```sh
python trainer.py
```

Provide the necessary arguments to train the model.
Run the following command to view help for the arguments

```sh
python trainer.py --help
```

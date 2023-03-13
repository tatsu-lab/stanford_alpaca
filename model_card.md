---
# Alpaca Model Card

## Model details
**Organization developing the model**
Stanford Hashimoto Group

**Model date**
Alpaca was trained in March 2023

**Model version**
This is version 1 of the model.

**Model type**
Alpaca models are instruction-following models finetuned from LLaMA models.

**More information**
Please see our blog post at `link` for more information.

**Citations details**
Please cite the [github repo](https://github.com/tatsu-lab/stanford_alpaca) if you use the data or code in this repo.

**License**
Code and data are licensed under the Apache 2.0 license.

**Where to send questions or comments about the model**
Questions and comments about LLaMA can be sent via the [GitHub repository](https://github.com/tatsu-lab/stanford_alpaca) of the project, by opening an issue.

## Intended use
**Primary intended uses**
The primary use of Alpaca is research on instruction following large language models.

**Primary intended users**
The primary intended users of the model are researchers in natural language processing, machine learning and artificial intelligence.

**Out-of-scope use cases**
Alpaca models are not finetuned with human feedback and are not intended for use in production systems.
Alpaca models are trained from data generated using the OpenAI API and thus any usage must not be competing with the OpenAI API.

## Metrics
**Model performance measures**
the Alpaca 7B model has been evaluated using blinded pairwise comparison with OpenAI's text-davinci-003 on the self-instruct evaluation set.
Our student authors have judged the Alpaca 7B model to be on par with text-davinci-003, with a win rate around 50%.

**Approaches to uncertainty and variability**
We have only finetuned a single Alpaca model at each model size, and thus we do not have a good sense of the variability of the model.

## Evaluation datasets
The model was evaluated on the self-instruct evaluation set.

## Training dataset
The model was trained on 52K instruction following data, which is release in the [Github repository](https://github.com/tatsu-lab/stanford_alpaca).
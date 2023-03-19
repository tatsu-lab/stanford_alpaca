from transformers import MarianMTModel, MarianTokenizer
from pprint import pprint
import json
from tqdm import tqdm
import json
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def translate(model, english_sentences=["I love to learn new things."]):
    # Tokenize the sentence
    inputs = tokenizer(english_sentences, return_tensors="pt", padding=True).to(DEVICE)

    # Translate the sentence
    outputs = model.generate(**inputs)
    spanish_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Print the translated sentence
    return spanish_sentence


def load_english_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return [[item["instruction"], item["input"], item["output"]] for item in data]


def flatten(data):
    return [item for sublist in data for item in sublist]


def process_batch(batch, empty_indices):
    translated_batch = translate(batch)
    for index, translated_item in zip(empty_indices, [""] * len(empty_indices)):
        translated_batch.insert(index, translated_item)
    return translated_batch


def translate_english_data(model, english_data):
    spanish_data = []
    batch, empty_indices = [], []

    for k, data in tqdm(enumerate(english_data), total=len(english_data)):
        if data == "":
            empty_indices.append(k)
            continue

        if k % batch_size == 0 and k != 0:
            spanish_data.extend(process_batch(batch, empty_indices))
            batch = []
            empty_indices = []

        batch.append(data)

    spanish_data.extend(process_batch(batch, empty_indices))
    return spanish_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default="data/English/English_train.json"
    )
    parser.add_argument(
        "--output_file", type=str, default="data/Spanish/Spanish_train.json"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-en-es")

    args = parser.parse_args()
    model_name = args.model_name
    model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    batch_size = args.batch_size

    english_data = flatten(load_english_data(args.input_file))
    spanish_data = translate_english_data(model, english_data)

    with open(args.output_file, "w") as f:
        json.dump(spanish_data, f)

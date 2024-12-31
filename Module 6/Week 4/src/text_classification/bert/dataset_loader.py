from datasets import load_dataset
from transformers import AutoTokenizer
from config import MAX_SEQ_LENGTH

model_name = "distilbert-base-uncased"  # bert-based-uncased

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

max_seq_length = min(MAX_SEQ_LENGTH, tokenizer.model_max_length)


def preprocess_function(examples):
    # Tokenize the texts

    result = tokenizer(
        examples["preprocessed_sentence"],
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
    )
    result["label"] = examples["label"]

    return result


def load_and_preprocess_dataset(preprocess_function):
    # Load dataset
    ds = load_dataset("thainq107/ntc-scv")
    # Running the preprocessing pipeline
    processed_dataset = ds.map(
        preprocess_function, batched=True, desc="Running tokenizer on dataset"
    )
    return processed_dataset

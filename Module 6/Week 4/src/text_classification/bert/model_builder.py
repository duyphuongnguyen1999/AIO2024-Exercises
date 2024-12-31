from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"


def build_model_and_tokenizer(model_name=model_name, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=num_labels,
        finetuning_task="text-classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    return model, tokenizer

import re
import string
import torchtext
from torchtext.vocab import build_vocab_from_iterator

torchtext.disable_torchtext_deprecation_warning()


# Preprocessing
def preprocess_text(text):
    # Remove URLs https:/www.
    url_pattern = re.compile(r"https?://\s+\wwww\.\s+")
    text = url_pattern.sub(r" ", text)

    # Remove HTML Tags: <>
    html_pattern = re.compile(r"<[^<>]+>")
    text = html_pattern.sub(" ", text)

    # Remove puncs and digits
    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, " ")

    # Remove emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F1F2-\U0001F1F4"  # Macau flag
        "\U0001F1E6-\U0001F1FF"  # flags
        "\U0001F600-\U0001F64F"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U0001F1F2"
        "\U0001F1F4"
        "\U0001F620"
        "\u200d"
        "\u2640-\u2642"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)

    # Normalize whitespace
    text = " ".join(text.split())

    # Lowercasing
    text = text.lower()
    return text


# Tokenizer and vocabulary creation
def yield_tokens(sentences, tokenizer):
    for sentence in sentences:
        yield tokenizer(sentence)


def build_vocabulary(dataset, tokenizer, vocab_size):
    vocabulary = build_vocab_from_iterator(
        yield_tokens(dataset, tokenizer),
        max_tokens=vocab_size,
        specials=["<pad>", "<unk>"],
    )
    vocabulary.set_default_index(vocabulary["<unk>"])
    return vocabulary


# Prepare dataset function
def prepare_dataset(dataset, vocabulary, tokenizer):
    # Create iterator for dataset: (sentence , label)
    for row in dataset:
        sentence = row["preprocessed_sentence"]
        encoded_sentence = vocabulary(tokenizer(sentence))
        label = row["label"]
        yield encoded_sentence, label

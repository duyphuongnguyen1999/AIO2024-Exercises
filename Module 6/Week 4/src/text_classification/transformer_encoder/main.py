# Import the necessary modules
import os
import torch
import torch.optim as optim
from datasets import load_dataset
from torchtext.data.functional import to_map_style_dataset
from torchtext.data import get_tokenizer
from model.train import train
from model.test import evaluate_test_dataloader
from data_processing.data_loader import create_dataloader
from data_processing.data_preprocessing import build_vocabulary
from data_processing.data_preprocessing import prepare_dataset
from model.Transformer import TransformerEncoderCls
from config import (
    VOCAB_SIZE,
    MAX_LENGTH,
    EMBED_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    FF_DIM,
    DROPOUT,
    LR,
    NUM_EPOCHS,
)


def main():
    # Load dataset
    ds = load_dataset("thainq107/ntc-scv")

    # Tokenizer
    tokenizer = get_tokenizer("basic_english")

    # Build vocabulary
    vocab_size = 10000
    vocabulary = build_vocabulary(
        dataset=ds["train"]["preprocessed_sentence"],
        tokenizer=tokenizer,
        vocab_size=vocab_size,
    )

    # Prepare dataset
    train_dataset = prepare_dataset(ds["train"], vocabulary, tokenizer)
    train_dataset = to_map_style_dataset(train_dataset)

    valid_dataset = prepare_dataset(ds["valid"], vocabulary, tokenizer)
    valid_dataset = to_map_style_dataset(valid_dataset)

    test_dataset = prepare_dataset(ds["test"], vocabulary, tokenizer)
    test_dataset = to_map_style_dataset(test_dataset)

    # Create DataLoader
    train_dataloader = create_dataloader(train_dataset, shuffle=True)
    valid_dataloader = create_dataloader(valid_dataset, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, shuffle=False)

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerEncoderCls(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        device=device,
    )
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0)

    save_model = "./trained_model"
    os.makedirs(save_model, exist_ok=True)
    model_name = "transformer_encoder"

    model, metrics = train(
        model=model,
        model_name=model_name,
        save_model=save_model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        num_epochs=NUM_EPOCHS,
        device=device,
    )

    print(metrics)

    evaluate_test_dataloader(
        model=model, test_dataloader=test_dataloader, device=device
    )


if __name__ == "__main__":
    main()

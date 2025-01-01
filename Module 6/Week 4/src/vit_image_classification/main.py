import torch
import os
from utils.dataset_loader import create_dataset_loader
from utils.train import train
from utils.test import evaluate_test_dataloader
from model.VisionTransformer import VisionTransformer
from config import IMG_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT, NUM_EPOCHS


def main():
    train_loader, val_loader, test_loader, classes, num_classes = (
        create_dataset_loader()
    )
    print(f"Classes: {classes}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(
        image_size=IMG_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        num_classes=num_classes,
        device=device,
    )
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0)

    save_model = "trained_model"
    os.makedirs(save_model, exist_ok=True)
    model_name = "vision_transformer_from_scratch"

    model, _ = train(
        model=model,
        model_name=model_name,
        save_model=save_model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
    )

    evaluate_test_dataloader(model=model, test_dataloader=test_loader, device=device)


if __name__ == "__main__":
    main()

from dataset_loader import load_and_preprocess_dataset, preprocess_function
from model_builder import build_model_and_tokenizer
from train_pipeline import train_model, compute_metrics


def main():
    # Load and preprocess dataset
    processed_dataset = load_and_preprocess_dataset(preprocess_function)

    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer()

    # Train model
    train_model(
        model=model,
        tokenizer=tokenizer,
        processed_dataset=processed_dataset,
        compute_metrics=compute_metrics,
    )


if __name__ == "__main__":
    main()

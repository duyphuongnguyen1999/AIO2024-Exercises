import torch
from sklearn.metrics import classification_report, accuracy_score


def predict(model, test_dataloader, device="cpu"):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Predict
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)  # Lấy lớp có xác suất cao nhất

            # Logging
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels


def evaluate_test_results(all_predictions, all_labels):
    # In báo cáo phân loại
    report = classification_report(all_labels, all_predictions, digits=3)
    print(report)

    # Tính accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Accuracy: {accuracy:.3f}")


def evaluate_test_dataloader(model, test_dataloader, device="cpu"):
    # Predict
    all_predictions, all_labels = predict(
        model=model, test_dataloader=test_dataloader, device=device
    )

    # Evaluate test result
    evaluate_test_results(all_predictions, all_labels)

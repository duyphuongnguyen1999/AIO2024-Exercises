import time
import torch
import matplotlib.pyplot as plt


# Train function for a single epoch
def train_epoch(
    model,
    optimizer,
    criterion,
    train_dataloader,
    device="cpu",
    epoch=0,
    log_interval=50,
):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        # Compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

        # Logging
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses) / len(losses)
            accuracy = total_acc / total_count
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f} "
                "| avg_loss {:8.3f} | time/batch {:8.3f}".format(
                    epoch,
                    idx,
                    len(train_dataloader),
                    accuracy,
                    avg_loss,
                    elapsed / log_interval,
                )
            )
            total_acc, total_count = 0, 0  # Reset accuracy for next interval
            start_time = time.time()  # Reset time for next interval

    # Return average accuracy and loss
    avg_loss = sum(losses) / len(losses)
    total_acc = total_acc / total_count if total_count > 0 else 0
    return total_acc, avg_loss


# Evalute
def evaluate_epoch(model, criterion, valid_dataloader, device="cpu"):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

            epoch_acc = total_acc / total_count
            epoch_loss = sum(losses) / len(losses)

            return epoch_acc, epoch_loss


# Train
def train(
    model,
    model_name,
    save_model,
    optimizer,
    criterion,
    train_dataloader,
    valid_dataloader,
    num_epochs,
    device="cpu",
):
    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100
    times = []
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        # Training
        train_acc, train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            device=device,
            epoch=epoch,
        )
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # Evaluation
        eval_acc, eval_loss = evaluate_epoch(
            model=model,
            criterion=criterion,
            valid_dataloader=valid_dataloader,
            device=device,
        )
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # Save best model
        if eval_loss < best_loss_eval:
            torch.save(model.state_dict(), save_model + f"/{model_name}.pt")

        times.append(time.time() - epoch_start_time)

        # Print lossm acc end epoch
        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f} | Train Accuracy {:8.3f} "
            "| Train Loss {:8.3f} | Valid Accuracy {:8.3f}"
            "| Valid Loss {:8.3f}".format(
                epoch,
                time.time() - epoch_start_time,
                train_acc,
                train_loss,
                eval_acc,
                eval_loss,
            )
        )
        print("-" * 59)

    # Load best model
    model.load_state_dict(torch.load(save_model + f"/{model_name}.pt"))
    model.eval()
    metrics = {
        "train_accuracy": train_accs,
        "train_loss": train_losses,
        "valid_accuracy": eval_accs,
        "valid_loss": eval_losses,
        "time": times,
    }
    return model, metrics


# Report
def plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].plot(epochs, train_accs, label="Training")
    axs[0].plot(epochs, eval_accs, label="Evaluation")
    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Loss")
    plt.legend()

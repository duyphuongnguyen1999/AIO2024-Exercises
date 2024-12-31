import torch
from torch.utils.data import DataLoader
from config import SEQ_LENGTH, BATCH_SIZE

# Set default seq_length, batch_size
seq_length = SEQ_LENGTH
batch_size = BATCH_SIZE


def collate_batch(batch):
    # Create inputs, offsetsm labels for batch
    sentences, labels = list(zip(*batch))
    encoded_sentences = [
        (
            # Add <pad> when len(sentence) < seq_length
            sentence + ([0] * (seq_length - len(sentence)))
            if len(sentence) < seq_length
            else sentence[:seq_length]
        )
        for sentence in sentences
    ]
    # Casting to torch.tensor
    encoded_sentences = torch.tensor(encoded_sentences, dtype=torch.int64)

    labels = torch.tensor(labels)

    return encoded_sentences, labels


def create_dataloader(dataset, shuffle=True):
    """
    Utility function to create a DataLoader with the provided dataset.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=0,
    )

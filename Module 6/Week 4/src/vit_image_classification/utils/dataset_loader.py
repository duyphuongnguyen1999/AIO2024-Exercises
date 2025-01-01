import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os
from config import TRAIN_RATIO, VALID_RATIO, IMG_SIZE, BATCH_SIZE


def create_dataset_loader():
    data_dir = "data"
    # Download dataset
    # import gdown
    # import zipfile
    # url = "https://drive.google.com/uc?id=1vSevps_hV5zhVf6aWuN8X7dd-qSAIgcc"
    # os.makedirs(data_dir, exist_ok=True)
    # gdown.download(url, output=data_dir, fuzzy=True)
    # Unzip
    # zip_path = os.path.join(data_dir, "flower_photos.zip")
    # extract_to = data_dir
    # with zipfile.ZipFile(zip_path, "r") as zip_ref:
    #     zip_ref.extractall(extract_to)

    # Load data
    data_patch = os.path.join(data_dir, "flower_photos")
    dataset = ImageFolder(root=data_patch)
    num_samples = len(dataset)
    classes = dataset.classes
    num_classes = len(dataset.classes)

    # Split
    n_train_examples = int(num_samples * TRAIN_RATIO)
    n_valid_examples = int(num_samples * VALID_RATIO)
    n_test_examples = num_samples - n_train_examples - n_valid_examples

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [n_train_examples, n_valid_examples, n_test_examples]
    )

    # Resize + Convert to tensor
    train_tranforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Apply transforms
    train_dataset.dataset.transform = train_tranforms
    valid_dataset.dataset.transform = test_transforms
    test_dataset.dataset.transform = test_transforms

    train_loader = DataLoader(
        dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE
    )

    val_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader, classes, num_classes

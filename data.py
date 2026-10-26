"""Real-image loading and architecture-batch reading."""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class ImageDataset(Dataset):
    """Read RGB images recursively, including flat COCO-style directories."""

    def __init__(self, root, image_size):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        self.paths = sorted(
            path
            for path in Path(root).rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not self.paths:
            raise ValueError(f"No images found in {root}.")
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            if min(image.size) < self.image_size:
                image = TF.resize(image, self.image_size)
            return self.transform(image)


class ArchitectureBatches:
    """Read numbered JSONL files containing [name, architecture] records."""

    def __init__(self, root, num_classes):
        self.paths = sorted(
            (path for path in Path(root).glob("*.jsonl") if path.stem.isdigit()),
            key=lambda path: int(path.stem),
        )
        if not self.paths:
            raise ValueError(f"No numbered architecture JSONL files found in {root}.")
        self.num_classes = num_classes

    def __getitem__(self, index):
        # Cycle explicitly when a run needs more architecture batches than available.
        path = self.paths[index % len(self.paths)]
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if len(records) < self.num_classes:
            raise ValueError(
                f"{path} must contain at least {self.num_classes} records."
            )
        architectures = [record[1] for record in records[: self.num_classes]]
        for architecture in architectures:
            if len(architecture) != 21:
                raise ValueError(f"{path}: expected 21 layer entries per architecture.")
        return architectures

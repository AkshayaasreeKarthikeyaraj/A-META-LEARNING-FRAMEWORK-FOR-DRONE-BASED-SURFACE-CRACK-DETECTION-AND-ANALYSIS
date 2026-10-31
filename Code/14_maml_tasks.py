
import random
import torch
import importlib

dataset_module = importlib.import_module("02_dataset")
CrackDataset = dataset_module.CrackDataset

preprocessing_module = importlib.import_module("03_preprocessing")
train_transform = preprocessing_module.train_transform


class MAMLTaskGenerator:

    def __init__(
        self,
        data_dir,
        dataset_name,
        support_size,
        query_size,
        seed=None,
        manifest_dir=None
    ):

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.support_size = support_size
        self.query_size = query_size
        self.manifest_dir = manifest_dir

        # Use one persistent RNG instead of recreating the same
        # random generator for every task.
        self.rng = random.Random(seed)

        # ---------------------------------------------------------
        # Manifest-based dataset
        # ---------------------------------------------------------
        if manifest_dir is not None:

            self.dataset = CrackDataset(
                data_dir=data_dir,
                split="train",
                dataset_name=dataset_name,
                transform=train_transform,
                manifest_dir=manifest_dir
            )

        # ---------------------------------------------------------
        # Original data_50 folder-based dataset
        # ---------------------------------------------------------
        else:

            self.dataset = CrackDataset(
                data_dir=data_dir,
                split="trn",
                dataset_name=dataset_name,
                transform=train_transform
            )

    def create_task(self):

        total_required = (
            self.support_size +
            self.query_size
        )

        if len(self.dataset) < total_required:

            raise ValueError(
                f"Not enough samples in {self.dataset_name}. "
                f"Required={total_required}, "
                f"Available={len(self.dataset)}"
            )

        indices = self.rng.sample(
            range(len(self.dataset)),
            total_required
        )

        support_indices = indices[
            :self.support_size
        ]

        query_indices = indices[
            self.support_size:
        ]

        support_images = []
        support_masks = []

        query_images = []
        query_masks = []

        for index in support_indices:

            image, mask = self.dataset[index]

            support_images.append(image)
            support_masks.append(mask)

        for index in query_indices:

            image, mask = self.dataset[index]

            query_images.append(image)
            query_masks.append(mask)

        support_images = torch.stack(
            support_images
        )

        support_masks = torch.stack(
            support_masks
        )

        query_images = torch.stack(
            query_images
        )

        query_masks = torch.stack(
            query_masks
        )

        return {
            "support_images": support_images,
            "support_masks": support_masks,
            "query_images": query_images,
            "query_masks": query_masks,
            "support_indices": support_indices,
            "query_indices": query_indices
        }

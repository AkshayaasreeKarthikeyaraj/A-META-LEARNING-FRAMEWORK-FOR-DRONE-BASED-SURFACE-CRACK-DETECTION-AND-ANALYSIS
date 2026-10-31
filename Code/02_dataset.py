
import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class CrackDataset(Dataset):

    def __init__(
        self,
        data_dir,
        split,
        dataset_name,
        transform=None,
        manifest_dir=None
    ):

        self.data_dir = data_dir
        self.split = split
        self.dataset_name = dataset_name
        self.transform = transform

        self.samples = []

        # ---------------------------------------------------------
        # 1. Check whether a manifest is being used
        # ---------------------------------------------------------

        if manifest_dir is not None:

            manifest_file = os.path.join(
                manifest_dir,
                dataset_name + ".json"
            )

            if not os.path.exists(manifest_file):
                raise FileNotFoundError(
                    f"Manifest not found: {manifest_file}"
                )

            with open(manifest_file, "r") as f:
                manifest = json.load(f)

            if split not in manifest:
                raise ValueError(
                    f"Split '{split}' not found in manifest: "
                    f"{manifest_file}"
                )

            for item in manifest[split]:

                image_path = item["image"]
                mask_path = item["mask"]

                if not os.path.exists(image_path):
                    raise FileNotFoundError(
                        f"Image not found: {image_path}"
                    )

                if not os.path.exists(mask_path):
                    raise FileNotFoundError(
                        f"Mask not found: {mask_path}"
                    )

                self.samples.append(
                    (image_path, mask_path)
                )

        # ---------------------------------------------------------
        # 2. Original folder-based dataset
        # ---------------------------------------------------------

        else:

            self.img_dir = os.path.join(
                data_dir,
                split,
                dataset_name,
                "imgs"
            )

            self.mask_dir = os.path.join(
                data_dir,
                split,
                dataset_name,
                "masks"
            )

            if not os.path.exists(self.img_dir):
                raise FileNotFoundError(
                    f"Image directory not found: {self.img_dir}"
                )

            if not os.path.exists(self.mask_dir):
                raise FileNotFoundError(
                    f"Mask directory not found: {self.mask_dir}"
                )

            image_files = sorted([
                f for f in os.listdir(self.img_dir)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ])

            for image_file in image_files:

                stem = os.path.splitext(image_file)[0]

                mask_file = None

                for ext in IMAGE_EXTENSIONS:

                    candidate = stem + ext

                    if os.path.exists(
                        os.path.join(
                            self.mask_dir,
                            candidate
                        )
                    ):

                        mask_file = candidate
                        break

                if mask_file is not None:

                    self.samples.append(
                        (
                            os.path.join(
                                self.img_dir,
                                image_file
                            ),
                            os.path.join(
                                self.mask_dir,
                                mask_file
                            )
                        )
                    )

        print(
            f"CrackDataset: {dataset_name} | "
            f"{split} | "
            f"{len(self.samples)} samples"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = np.array(
            Image.open(image_path).convert("RGB")
        )

        mask = np.array(
            Image.open(mask_path).convert("L")
        )

        mask = (mask >= 128).astype(np.uint8)

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

        else:

            image = torch.from_numpy(
                image.transpose(2, 0, 1)
            ).float() / 255.0

            mask = torch.from_numpy(
                mask
            ).float()

        if not torch.is_tensor(mask):

            mask = torch.from_numpy(mask)

        mask = mask.float()

        if mask.ndim == 2:

            mask = mask.unsqueeze(0)

        return image, mask

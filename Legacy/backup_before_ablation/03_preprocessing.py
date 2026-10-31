import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 256

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Training: resize + light augmentation + ImageNet normalisation
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# Validation / test: NO augmentation (must stay deterministic)
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

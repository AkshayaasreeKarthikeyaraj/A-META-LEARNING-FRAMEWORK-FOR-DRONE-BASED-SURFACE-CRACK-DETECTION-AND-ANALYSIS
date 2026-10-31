
import os
import sys
import torch
import torch.optim as optim
import importlib

PROJECT_DIR = "/content/drive/MyDrive/Drone_Wall_Crack"
DATA_DIR = os.path.join(PROJECT_DIR, "data_50")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if CODE_DIR := os.path.dirname(os.path.abspath(__file__)):
    if CODE_DIR not in sys.path:
        sys.path.append(CODE_DIR)

dataset_module = importlib.import_module("02_dataset")
model_module = importlib.import_module("12_model")
loss_module = importlib.import_module("11_new_loss")

CrackDataset = dataset_module.CrackDataset
CrackAnalysisModel = model_module.CrackAnalysisModel
HybridLoss = loss_module.HybridLoss

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 1
LEARNING_RATE = 1e-4

DATASETS = [
    "Volker",
    "Crack500",
    "OUC"
]

model = CrackAnalysisModel().to(DEVICE)

criterion = HybridLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_loss = float("inf")

history = {
    "train_loss": [],
    "val_loss": []
}


def run_epoch(dataset, training=True):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for index in range(len(dataset)):

        image, mask = dataset[index]

        image = image.unsqueeze(0).to(DEVICE)
        mask = mask.unsqueeze(0).to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            output = model(image)

            losses = criterion(
                output["segmentation_logits"],
                output["coarse_map"],
                output["refined_map"],
                mask
            )

            loss = losses["total_loss"]

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset)


for epoch in range(1, EPOCHS + 1):

    train_loss = 0.0
    val_loss = 0.0

    for dataset_name in DATASETS:

        train_dataset = CrackDataset(
            data_dir=DATA_DIR,
            split="trn",
            dataset_name=dataset_name
        )

        val_dataset = CrackDataset(
            data_dir=DATA_DIR,
            split="val",
            dataset_name=dataset_name
        )

        dataset_train_loss = run_epoch(
            train_dataset,
            training=True
        )

        dataset_val_loss = run_epoch(
            val_dataset,
            training=False
        )

        train_loss += dataset_train_loss
        val_loss += dataset_val_loss

        print(
            "Epoch",
            epoch,
            "|",
            dataset_name,
            "| Train:",
            round(dataset_train_loss, 4),
            "| Val:",
            round(dataset_val_loss, 4)
        )

    train_loss /= len(DATASETS)
    val_loss /= len(DATASETS)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(
        "Epoch",
        epoch,
        "| Average Train:",
        round(train_loss, 4),
        "| Average Val:",
        round(val_loss, 4)
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "history": history
    }

    torch.save(
        checkpoint,
        os.path.join(
            CHECKPOINT_DIR,
            "latest_checkpoint.pth"
        )
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            checkpoint,
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
        )

        print("Best model saved")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss
    },
    os.path.join(
        CHECKPOINT_DIR,
        "trained_model_final.pth"
    )
)

print()
print("Training completed")
print("Best validation loss:", best_val_loss)
print(
    "Final model:",
    os.path.join(
        CHECKPOINT_DIR,
        "trained_model_final.pth"
    )
)

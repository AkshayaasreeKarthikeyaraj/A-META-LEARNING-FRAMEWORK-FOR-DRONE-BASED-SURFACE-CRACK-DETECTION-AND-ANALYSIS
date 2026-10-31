import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader

_maml = importlib.import_module("17_maml_train")


class SupervisedTrainer(_maml.MAMLTrainer):
    """
    Ordinary (non-meta) training of the same network with the same loss, optimiser and schedule.
    One epoch = `updates_per_epoch` optimiser steps per data set (same number of updates as the
    MAML trainer). Model selection = validation Dice WITHOUT test-time adaptation.
    """

    def __init__(self, *args, batch_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self._iters = {}

    def _batches(self, name):
        CrackDataset = self.dataset_module.CrackDataset
        ds = CrackDataset(
            self.data_dir, "train", name,
            transform=self.prep_module.train_transform,
            manifest_dir=self.split_manifest_dir,
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            drop_last=True, num_workers=0)
        while True:
            for batch in loader:
                yield batch

    def train_one_epoch(self, active_datasets=None):

        self.model.train()
        active = [d for d in (active_datasets or self.DATASETS) if d in self.DATASETS]
        results = {}

        for name in active:

            if name not in self._iters:
                self._iters[name] = self._batches(name)
            it = self._iters[name]
            loss_fn = self.loss_functions[name]
            log = []

            for _ in range(self.updates_per_epoch):

                images, masks = next(it)
                images = images.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(images)
                loss = loss_fn(
                    out["segmentation_logits"], out["coarse_map"],
                    out["refined_map"], masks,
                )["total_loss"]

                if not torch.isfinite(loss):
                    continue

                loss.backward()
                if self.meta_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.meta_grad_clip)
                self.optimizer.step()
                log.append(float(loss.detach()))

            results[name] = {
                "support_loss": float("nan"),
                "query_loss": float(np.mean(log)) if log else float("nan"),
            }
            torch.cuda.empty_cache()

        return {"datasets": results}

    def validate(self, name):
        """Validation Dice with NO adaptation."""
        _, _, loader = self._val_bundle(name)
        self.model.eval()
        counts = {t: [0, 0, 0] for t in self.VAL_THRESHOLDS}

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                probs = torch.sigmoid(self.model(images)["segmentation_logits"])
                target = masks >= 0.5
                for t in self.VAL_THRESHOLDS:
                    pred = probs >= t
                    counts[t][0] += (pred & target).sum().item()
                    counts[t][1] += (pred & ~target).sum().item()
                    counts[t][2] += ((~pred) & target).sum().item()

        dice = {t: 2 * c[0] / (2 * c[0] + c[1] + c[2] + 1e-8) for t, c in counts.items()}
        best_t = max(dice, key=dice.get)
        self.model.train()
        return {"val_dice": dice[best_t], "val_threshold": best_t, "dice_by_threshold": dice}

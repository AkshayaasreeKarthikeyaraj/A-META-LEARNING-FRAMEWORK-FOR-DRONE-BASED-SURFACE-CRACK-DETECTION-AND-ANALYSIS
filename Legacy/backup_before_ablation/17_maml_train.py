import os
import random
import importlib

import numpy as np
import torch
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset


class MAMLTrainer:
    """
    MAML trainer (v2).

    One epoch  = `updates_per_epoch` meta-updates per dataset.
    One update = average query loss of `meta_batch_size` tasks.
    Model selection = validation Dice (after adapting on a support set),
                      NOT the training query loss.
    """

    DATASETS = ["Volker", "Crack500", "OUC"]
    VAL_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def __init__(
        self,
        model,
        data_dir,
        device,
        checkpoint_dir,
        split_name,
        manifest_dir,
        meta_lr=1e-4,
        inner_lr=1e-2,
        weight_decay=1e-5,
        meta_batch_size=2,
        updates_per_epoch=20,
        second_order=False,
        meta_grad_clip=5.0,
        inner_grad_clip=5.0,
        val_max_images=96,
        update_steps=None,
        support_sizes=None,
        query_sizes=None,
        pos_weights=None,
    ):
        self.model = model
        self.data_dir = data_dir
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.split_name = split_name
        self.manifest_dir = manifest_dir
        self.split_manifest_dir = os.path.join(manifest_dir, split_name)
        if not os.path.exists(self.split_manifest_dir):
            raise FileNotFoundError(self.split_manifest_dir)

        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.meta_batch_size = meta_batch_size
        self.updates_per_epoch = updates_per_epoch
        self.second_order = second_order
        self.meta_grad_clip = meta_grad_clip
        self.inner_grad_clip = inner_grad_clip
        self.val_max_images = val_max_images

        self.update_steps = update_steps or {"Volker": 3, "Crack500": 3, "OUC": 3}
        self.support_sizes = support_sizes or {"Volker": 4, "Crack500": 4, "OUC": 5}
        self.query_sizes = query_sizes or {"Volker": 4, "Crack500": 4, "OUC": 5}
        self.pos_weights = pos_weights or {"Volker": 4.0, "Crack500": 2.0, "OUC": 4.0}

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=meta_lr, weight_decay=weight_decay
        )

        self.inner_module = importlib.import_module("15_maml_inner_loop")
        self.loss_module = importlib.import_module("11_new_loss")
        self.task_module = importlib.import_module("14_maml_tasks")
        self.dataset_module = importlib.import_module("02_dataset")
        self.prep_module = importlib.import_module("03_preprocessing")

        self.loss_functions = {
            name: self.loss_module.HybridLoss(
                pos_weight=self.pos_weights[name],
                lambda_coarse=1.0,
                lambda_refine=1.0,
            ).to(device)
            for name in self.DATASETS
        }

        # one task generator per dataset (built once, reused every task)
        self.generators = {
            name: self.task_module.MAMLTaskGenerator(
                data_dir=data_dir,
                dataset_name=name,
                support_size=self.support_sizes[name],
                query_size=self.query_sizes[name],
                manifest_dir=self.split_manifest_dir,
            )
            for name in self.DATASETS
        }

        self._val_cache = {}

    # ------------------------------------------------------------
    def create_task(self, dataset_name):
        task = self.generators[dataset_name].create_task()
        return {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in task.items()
        }

    def make_inner(self, dataset_name):
        return self.inner_module.MAMLInnerLoop(
            self.model,
            self.loss_functions[dataset_name],
            inner_lr=self.inner_lr,
            update_steps=self.update_steps[dataset_name],
            second_order=self.second_order,
            grad_clip=self.inner_grad_clip,
        )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------
    def train_one_epoch(self, active_datasets=None):

        self.model.train()
        active = [d for d in (active_datasets or self.DATASETS) if d in self.DATASETS]
        results = {}

        for name in active:

            loss_fn = self.loss_functions[name]
            inner = self.make_inner(name)
            sup_log, qry_log = [], []

            for _ in range(self.updates_per_epoch):

                self.optimizer.zero_grad(set_to_none=True)
                used = 0

                for _ in range(self.meta_batch_size):

                    task = self.create_task(name)

                    adapted, support_loss = inner.adapt(
                        task["support_images"], task["support_masks"]
                    )

                    out = functional_call(
                        self.model, adapted, (task["query_images"],)
                    )
                    query_loss = loss_fn(
                        out["segmentation_logits"],
                        out["coarse_map"],
                        out["refined_map"],
                        task["query_masks"],
                    )["total_loss"]

                    if not torch.isfinite(query_loss):
                        continue

                    (query_loss / self.meta_batch_size).backward()
                    used += 1

                    sup_log.append(float(support_loss.detach()))
                    qry_log.append(float(query_loss.detach()))

                    del task, adapted, out, query_loss, support_loss

                if used == 0:
                    continue

                if self.meta_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.meta_grad_clip
                    )

                self.optimizer.step()

            results[name] = {
                "support_loss": float(np.mean(sup_log)) if sup_log else float("nan"),
                "query_loss": float(np.mean(qry_log)) if qry_log else float("nan"),
            }

            torch.cuda.empty_cache()

        return {"datasets": results}

    # ------------------------------------------------------------
    # VALIDATE  (adapt on fixed support set, evaluate on val split)
    # ------------------------------------------------------------
    def _val_bundle(self, name):

        if name in self._val_cache:
            return self._val_cache[name]

        CrackDataset = self.dataset_module.CrackDataset
        vt = self.prep_module.val_transform

        train_ds = CrackDataset(
            self.data_dir, "train", name,
            transform=vt, manifest_dir=self.split_manifest_dir,
        )
        idx = random.Random(1234).sample(range(len(train_ds)), self.support_sizes[name])
        s_imgs = torch.stack([train_ds[i][0] for i in idx]).to(self.device)
        s_msks = torch.stack([train_ds[i][1] for i in idx]).to(self.device)

        val_ds = CrackDataset(
            self.data_dir, "val", name,
            transform=vt, manifest_dir=self.split_manifest_dir,
        )
        n = len(val_ds)
        k = min(n, self.val_max_images)
        sel = np.linspace(0, n - 1, k).astype(int).tolist()
        loader = DataLoader(Subset(val_ds, sel), batch_size=8, shuffle=False, num_workers=0)

        self._val_cache[name] = (s_imgs, s_msks, loader)
        return self._val_cache[name]

    def validate(self, name):

        s_imgs, s_msks, loader = self._val_bundle(name)

        # BN running stats must not be polluted by validation passes
        buffers_backup = {n: b.clone() for n, b in self.model.named_buffers()}

        self.model.train()
        inner = self.make_inner(name)
        adapted, _ = inner.adapt(s_imgs, s_msks)
        adapted = {n: p.detach() for n, p in adapted.items()}

        self.model.eval()

        counts = {t: [0, 0, 0] for t in self.VAL_THRESHOLDS}   # tp, fp, fn

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                out = functional_call(self.model, adapted, (images,))
                probs = torch.sigmoid(out["segmentation_logits"])
                target = masks >= 0.5
                for t in self.VAL_THRESHOLDS:
                    pred = probs >= t
                    counts[t][0] += (pred & target).sum().item()
                    counts[t][1] += (pred & ~target).sum().item()
                    counts[t][2] += ((~pred) & target).sum().item()

        dice = {
            t: 2 * c[0] / (2 * c[0] + c[1] + c[2] + 1e-8)
            for t, c in counts.items()
        }
        best_t = max(dice, key=dice.get)

        with torch.no_grad():
            for n, b in self.model.named_buffers():
                b.copy_(buffers_backup[n])

        self.model.train()

        return {"val_dice": dice[best_t], "val_threshold": best_t, "dice_by_threshold": dice}

    # ------------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------------
    def save_checkpoint(self, epoch, filename, extra_state=None):

        path = os.path.join(self.checkpoint_dir, filename)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "split_name": self.split_name,
            # hyper-parameters, so evaluation uses exactly the same ones
            "inner_lr": self.inner_lr,
            "update_steps": self.update_steps,
            "support_sizes": self.support_sizes,
            "pos_weights": self.pos_weights,
            "second_order": self.second_order,
        }
        if extra_state:
            ckpt.update(extra_state)

        torch.save(ckpt, path)
        return path

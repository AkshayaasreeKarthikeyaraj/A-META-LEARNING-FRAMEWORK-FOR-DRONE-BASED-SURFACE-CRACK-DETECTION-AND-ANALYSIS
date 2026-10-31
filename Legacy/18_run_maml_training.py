
import os
import sys
import torch
import importlib
import gc
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = "/content/drive/MyDrive/Drone_Wall_Crack"

CODE_DIR = os.path.join(
    PROJECT_DIR,
    "code"
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data_50"
)

MANIFEST_DIR = os.path.join(
    PROJECT_DIR,
    "split_manifests"
)


# ============================================================
# SPLIT
# ============================================================

SPLIT_NAME = "70_15_15"

SPLIT_MANIFEST_DIR = os.path.join(
    MANIFEST_DIR,
    SPLIT_NAME
)


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    SPLIT_NAME
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# PYTHON PATH
# ============================================================

if CODE_DIR not in sys.path:

    sys.path.insert(
        0,
        CODE_DIR
    )

importlib.invalidate_caches()


# ============================================================
# IMPORT MODULES
# ============================================================

model_module = importlib.import_module(
    "12_model"
)

loss_module = importlib.import_module(
    "11_new_loss"
)

train_module = importlib.import_module(
    "17_maml_train"
)

CrackAnalysisModel = (
    model_module.CrackAnalysisModel
)

HybridLoss = (
    loss_module.HybridLoss
)

MAMLTrainer = (
    train_module.MAMLTrainer
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

MAX_EPOCHS = 100

PATIENCE = 10

MIN_DELTA = 0.001

DATASETS = [
    "Volker",
    "Crack500",
    "OUC"
]


# ============================================================
# VALIDATE SPLIT
# ============================================================

VALID_SPLITS = [
    "60_20_20",
    "70_15_15",
    "80_10_10"
]

if SPLIT_NAME not in VALID_SPLITS:

    raise ValueError(
        f"Invalid split: {SPLIT_NAME}"
    )


if not os.path.exists(
    SPLIT_MANIFEST_DIR
):

    raise FileNotFoundError(
        "Manifest directory not found:\n"
        f"{SPLIT_MANIFEST_DIR}"
    )


# ============================================================
# CREATE MODEL
# ============================================================

model = CrackAnalysisModel().to(
    DEVICE
)

criterion = HybridLoss()


# ============================================================
# CREATE TRAINER
# ============================================================

trainer = MAMLTrainer(
    model=model,
    loss_function=criterion,
    data_dir=DATA_DIR,
    device=DEVICE,
    checkpoint_dir=CHECKPOINT_DIR,
    meta_lr=1e-5,
    inner_lr=1e-3,
    weight_decay=1e-8,
    momentum=0.999,
    split_name=SPLIT_NAME,
    manifest_dir=MANIFEST_DIR
)


# ============================================================
# TRAINING INFORMATION
# ============================================================

TRAINING_SAMPLES = {
    "Volker": 693,
    "Crack500": 2354,
    "OUC": 903
}


# ============================================================
# PRINT HEADER
# ============================================================

print()
print("=" * 76)
print("                 MAML CRACK SEGMENTATION")
print("=" * 76)
print()

print(f"Device          : {DEVICE}")
print(f"Split           : {SPLIT_NAME}")
print(f"Maximum epochs  : {MAX_EPOCHS}")
print(f"Patience        : {PATIENCE} epochs per dataset")
print(f"Minimum delta   : {MIN_DELTA}")
print()

print("Training samples")
print("-" * 76)

for dataset_name in DATASETS:

    print(
        f"  {dataset_name:<14}: "
        f"{TRAINING_SAMPLES[dataset_name]}"
    )

print()

print("Early stopping strategy")
print("-" * 76)
print("  Each dataset has its own patience counter.")
print("  A stopped dataset is skipped.")
print("  Other datasets continue training.")
print("  Training ends only when all datasets stop.")
print()

print("Checkpoint path")
print("-" * 76)
print(CHECKPOINT_DIR)
print()

print("=" * 76)
print()


# ============================================================
# PER-DATASET TRAINING STATE
# ============================================================

state = {}

for dataset_name in DATASETS:

    state[dataset_name] = {

        "best_query_loss":
            float("inf"),

        "best_epoch":
            0,

        "epochs_without_improvement":
            0,

        "stopped":
            False,

        "history":
            [],

        "best_checkpoint":
            None
    }


# ============================================================
# GLOBAL HISTORY
# ============================================================

history = []


# ============================================================
# TRAINING START
# ============================================================

print("STARTING TRAINING")
print("=" * 76)
print()


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    # ========================================================
    # FIND ACTIVE DATASETS
    # ========================================================

    active_datasets = [
        dataset_name
        for dataset_name in DATASETS
        if not state[dataset_name]["stopped"]
    ]


    # ========================================================
    # ALL DATASETS STOPPED
    # ========================================================

    if len(active_datasets) == 0:

        print()
        print("=" * 76)
        print("ALL DATASETS HAVE STOPPED")
        print("=" * 76)

        break


    # ========================================================
    # EPOCH HEADER
    # ========================================================

    print()
    print("=" * 76)
    print(
        f"EPOCH {epoch:03d}/{MAX_EPOCHS:03d}"
    )
    print("=" * 76)
    print()


    print(
        f"Active datasets: "
        f"{len(active_datasets)}/{len(DATASETS)}"
    )

    print()


    # ========================================================
    # TRAIN ACTIVE DATASETS
    # ========================================================

    results = trainer.train_one_epoch(
        active_datasets=active_datasets
    )

    dataset_results = results[
        "datasets"
    ]


    # ========================================================
    # CURRENT EPOCH RECORD
    # ========================================================

    epoch_record = {
        "epoch": epoch,
        "datasets": {}
    }


    # ========================================================
    # PROCESS EACH DATASET
    # ========================================================

    for dataset_name in DATASETS:

        dataset_state = state[
            dataset_name
        ]


        # ----------------------------------------------------
        # ALREADY STOPPED
        # ----------------------------------------------------

        if dataset_state["stopped"]:

            dataset_state["history"].append({

                "epoch": epoch,

                "support_loss": None,

                "query_loss": None,

                "status": "STOPPED"

            })

            epoch_record[
                "datasets"
            ][dataset_name] = {

                "support_loss": None,

                "query_loss": None,

                "best_query_loss":
                    dataset_state[
                        "best_query_loss"
                    ],

                "patience":
                    dataset_state[
                        "epochs_without_improvement"
                    ],

                "status": "STOPPED"

            }

            continue


        # ----------------------------------------------------
        # GET RESULT
        # ----------------------------------------------------

        dataset_result = dataset_results[
            dataset_name
        ]

        support_loss = float(
            dataset_result[
                "support_loss"
            ]
        )

        query_loss = float(
            dataset_result[
                "query_loss"
            ]
        )


        # ----------------------------------------------------
        # CHECK IMPROVEMENT
        # ----------------------------------------------------

        improved = (
            query_loss
            <
            (
                dataset_state[
                    "best_query_loss"
                ]
                -
                MIN_DELTA
            )
        )


        if improved:

            dataset_state[
                "best_query_loss"
            ] = query_loss

            dataset_state[
                "best_epoch"
            ] = epoch

            dataset_state[
                "epochs_without_improvement"
            ] = 0


            # ------------------------------------------------
            # DATASET-SPECIFIC BEST CHECKPOINT
            # ------------------------------------------------

            best_filename = (
                "maml_best_"
                f"{dataset_name}"
                "_checkpoint.pth"
            )

            best_path = (
                trainer.save_checkpoint(
                    epoch=epoch,

                    support_loss=support_loss,

                    query_loss=query_loss,

                    filename=best_filename,

                    extra_state={
                        "dataset_name":
                            dataset_name,

                        "best_epoch":
                            epoch,

                        "best_query_loss":
                            query_loss
                    }
                )
            )

            dataset_state[
                "best_checkpoint"
            ] = best_path

            status = "NEW BEST"


        else:

            dataset_state[
                "epochs_without_improvement"
            ] += 1

            status = (
                "NO IMPROVEMENT"
            )


        # ----------------------------------------------------
        # CHECK DATASET EARLY STOPPING
        # ----------------------------------------------------

        if (
            dataset_state[
                "epochs_without_improvement"
            ]
            >= PATIENCE
        ):

            dataset_state[
                "stopped"
            ] = True

            status = "STOPPED"


        # ----------------------------------------------------
        # SAVE DATASET HISTORY
        # ----------------------------------------------------

        dataset_state[
            "history"
        ].append({

            "epoch": epoch,

            "support_loss":
                support_loss,

            "query_loss":
                query_loss,

            "status":
                status
        })


        # ----------------------------------------------------
        # EPOCH RECORD
        # ----------------------------------------------------

        epoch_record[
            "datasets"
        ][dataset_name] = {

            "support_loss":
                support_loss,

            "query_loss":
                query_loss,

            "best_query_loss":
                dataset_state[
                    "best_query_loss"
                ],

            "patience":
                dataset_state[
                    "epochs_without_improvement"
                ],

            "status":
                status
        }


    # ========================================================
    # SAVE GLOBAL HISTORY
    # ========================================================

    history.append(
        epoch_record
    )


    # ========================================================
    # PRINT STRUCTURED DATASET RESULTS
    # ========================================================

    print("-" * 76)

    for dataset_name in DATASETS:

        dataset_state = state[
            dataset_name
        ]

        dataset_record = epoch_record[
            "datasets"
        ][dataset_name]


        if (
            dataset_record[
                "query_loss"
            ]
            is None
        ):

            print(
                f"{dataset_name:<12} | "
                f"Query: {'--':>8} | "
                f"Best: "
                f"{dataset_state['best_query_loss']:.4f} | "
                f"Patience: "
                f"{dataset_state['epochs_without_improvement']}"
                f"/{PATIENCE} | "
                f"STOPPED"
            )

        else:

            print(
                f"{dataset_name:<12} | "
                f"Query: "
                f"{dataset_record['query_loss']:>8.4f} | "
                f"Best: "
                f"{dataset_state['best_query_loss']:>8.4f} | "
                f"Patience: "
                f"{dataset_state['epochs_without_improvement']}"
                f"/{PATIENCE} | "
                f"{dataset_record['status']}"
            )


    print("-" * 76)

    active_count = sum(
        not state[dataset_name]["stopped"]
        for dataset_name in DATASETS
    )

    print(
        f"Active datasets: "
        f"{active_count}/{len(DATASETS)}"
    )


    # ========================================================
    # SAVE LATEST CHECKPOINT
    # ========================================================

    latest_path = trainer.save_checkpoint(

        epoch=epoch,

        support_loss=None,

        query_loss=None,

        filename="maml_latest_checkpoint.pth",

        extra_state={

            "dataset_states":
                state,

            "history":
                history
        }
    )


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_path = os.path.join(
        CHECKPOINT_DIR,
        "maml_training_history.pth"
    )

    torch.save(
        {
            "history": history,
            "dataset_states": state,
            "split_name": SPLIT_NAME
        },
        history_path
    )


    # ========================================================
    # CHECK IF ALL DATASETS STOPPED
    # ========================================================

    if all(
        state[dataset_name]["stopped"]
        for dataset_name in DATASETS
    ):

        print()
        print("=" * 76)
        print(
            f"ALL DATASETS STOPPED AT EPOCH {epoch}"
        )
        print("=" * 76)

        break


# ============================================================
# FINAL EPOCH
# ============================================================

if len(history) > 0:

    final_epoch = history[-1][
        "epoch"
    ]

else:

    final_epoch = 0


# ============================================================
# FINAL CHECKPOINT
# ============================================================

final_path = trainer.save_checkpoint(

    epoch=final_epoch,

    support_loss=None,

    query_loss=None,

    filename="maml_final_checkpoint.pth",

    extra_state={

        "dataset_states":
            state,

        "history":
            history
    }
)


# ============================================================
# SAVE FINAL HISTORY
# ============================================================

history_path = os.path.join(
    CHECKPOINT_DIR,
    "maml_training_history.pth"
)

torch.save(

    {
        "history": history,

        "dataset_states":
            state,

        "split_name":
            SPLIT_NAME,

        "final_epoch":
            final_epoch

    },

    history_path
)


# ============================================================
# GRAPH 1
# ALL DATASETS QUERY LOSS
# ============================================================

plt.figure(
    figsize=(11, 7)
)

for dataset_name in DATASETS:

    epochs = []

    query_losses = []

    for record in state[
        dataset_name
    ]["history"]:

        if (
            record["query_loss"]
            is not None
        ):

            epochs.append(
                record["epoch"]
            )

            query_losses.append(
                record["query_loss"]
            )

    if len(epochs) > 0:

        plt.plot(
            epochs,
            query_losses,
            linewidth=2,
            label=dataset_name
        )


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Query Loss"
)

plt.title(
    f"{SPLIT_NAME} MAML - Query Loss by Dataset"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


graph_path = os.path.join(
    CHECKPOINT_DIR,
    "maml_query_loss_all_datasets.png"
)

plt.savefig(
    graph_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# INDIVIDUAL DATASET GRAPHS
# ============================================================

dataset_graph_paths = {}

for dataset_name in DATASETS:

    epochs = []

    query_losses = []

    for record in state[
        dataset_name
    ]["history"]:

        if (
            record["query_loss"]
            is not None
        ):

            epochs.append(
                record["epoch"]
            )

            query_losses.append(
                record["query_loss"]
            )


    plt.figure(
        figsize=(10, 6)
    )

    if len(epochs) > 0:

        plt.plot(
            epochs,
            query_losses,
            linewidth=2,
            label=dataset_name
        )


    best_epoch = state[
        dataset_name
    ]["best_epoch"]

    best_loss = state[
        dataset_name
    ]["best_query_loss"]


    if best_epoch > 0:

        plt.scatter(
            [best_epoch],
            [best_loss],
            s=70,
            label=(
                f"Best "
                f"(Epoch {best_epoch})"
            )
        )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Query Loss"
    )

    plt.title(
        f"{SPLIT_NAME} MAML - "
        f"{dataset_name} Query Loss"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    dataset_graph_path = os.path.join(
        CHECKPOINT_DIR,
        (
            "maml_query_loss_"
            f"{dataset_name}.png"
        )
    )

    plt.savefig(
        dataset_graph_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    dataset_graph_paths[
        dataset_name
    ] = dataset_graph_path


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 76)
print("                     TRAINING COMPLETED")
print("=" * 76)
print()

print(
    f"Split            : {SPLIT_NAME}"
)

print(
    f"Epochs completed : {final_epoch}"
)

print()

print("DATASET SUMMARY")
print("-" * 76)

for dataset_name in DATASETS:

    dataset_state = state[
        dataset_name
    ]

    print()
    print(
        f"{dataset_name}"
    )

    print(
        f"  Best epoch       : "
        f"{dataset_state['best_epoch']}"
    )

    print(
        f"  Best query loss  : "
        f"{dataset_state['best_query_loss']:.4f}"
    )

    print(
        f"  Final patience   : "
        f"{dataset_state['epochs_without_improvement']}"
        f"/{PATIENCE}"
    )

    print(
        f"  Status            : "
        f"{'STOPPED' if dataset_state['stopped'] else 'ACTIVE'}"
    )

    print(
        f"  Best checkpoint   : "
        f"{dataset_state['best_checkpoint']}"
    )


print()
print("SAVED FILES")
print("-" * 76)

print()
print(
    "Latest checkpoint:"
)

print(
    latest_path
)

print()
print(
    "Final checkpoint:"
)

print(
    final_path
)

print()
print(
    "Training history:"
)

print(
    history_path
)

print()
print(
    "Combined query-loss graph:"
)

print(
    graph_path
)

print()

for dataset_name in DATASETS:

    print(
        f"{dataset_name} query-loss graph:"
    )

    print(
        dataset_graph_paths[
            dataset_name
        ]

    )

print()
print("=" * 76)

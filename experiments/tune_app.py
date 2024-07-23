# experiments/tune_app.py

import logging
from functools import partial
from typing import Any, Dict

import lightning as L
import psutil
import torch
import torch_geometric.transforms as T
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.bayesopt import BayesOptSearch

logging_config()


def train_tune(
    config: Dict[str, Any],
    callback: TuneReportCallback,
    train_data: HeteroData,
    val_data: HeteroData,
    num_proc: int,
    accelerator: str,
):
    logging.info("Creating train dataloader...")
    # TODO

    logging.info("Creating validation dataloader...")
    # TODO

    logging.info("Creating model...")
    # TODO

    logging.info("Training and testing model...")
    trainer = L.Trainer(
        accelerator=accelerator, devices=-1, max_epochs=10, callbacks=[callback]
    )
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    args = TuneLinkPredictionArgparse.parse_known_args()

    num_proc = (
        psutil.cpu_count(logical=False) if args.num_proc is None else args.num_proc
    )
    logging.info(f"Number of processes: {num_proc}.")

    if args.accelerator is None:
        if torch.cuda.is_available():
            accelerator = "gpu"
        else:
            accelerator = "cpu"
    else:
        accelerator = args.accelerator
    logging.info(f"Accelerator: {accelerator}.")

    logging.info("Creating train, validation, and test datasets...")
    # TODO

    callback = TuneReportCallback(
        {"loss": "val_loss", "roc_auc": "val_roc_auc"}, on="validation_end"
    )

    # TODO: Tune
    config = {
        "num_neighbors": tune.choice([16, 32, 64, 128]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "hidden_channels": tune.choice([16, 32, 64, 128]),
        "dropout": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-4, 0.0),
    }
    tune.run(
        partial(
            train_tune,
            callback=callback,
            train_data=train_data,
            val_data=val_data,
            num_proc=num_proc,
            accelerator=accelerator,
        ),
        metric="roc_auc",
        mode="max",
        name=f"Tune Link Prediction {GNN}",
        resources_per_trial={"cpu": num_proc, "gpu": 1},
        num_samples=10,
        search_alg=BayesOptSearch(),
        scheduler=AsyncHyperBandScheduler(),
        config=config,
    )

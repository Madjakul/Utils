# main.py

import logging

import lightning as L
import psutil
import torch

logging_config()


if __name__ == "__main__":
    args = LinkPredictionArgparse.parse_known_args()

    config = helpers.load_config_from_file(args.config_file)
    logging.info(f"Configuration file loaded from {args.config_file}.")

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

    if args.wandb:
        wandb_logger = WandbLogger(
            project="HALvest-Geometric",
            name=f"link-prediction-{config['gnn']}-{args.run}",
        )
        logging.info("WandB logger enabled.")

    logging.info("Creating train, validation, and test datasets...")
    # TODO

    logging.info("Creating train dataloader...")
    # TODO

    logging.info("Creating validation dataloader...")
    # TODO

    logging.info("Creating test dataloader...")
    # TODO

    logging.info("Creating model...")
    # TODO

    logging.info("Training and testing model...")
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=-1,
        logger=wandb_logger if args.wandb else None,
        max_epochs=config["max_epochs"],
    )
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.test(model=model, dataloaders=test_dataloader)

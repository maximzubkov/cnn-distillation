from argparse import ArgumentParser
from os.path import join

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import WandbLogger

from configs import (
    get_resnet_student_config,
    get_resnet_teacher_config,
    get_test_hyperparams,
    get_default_hyperparams,
    get_distillation_config,
)
from models import SingleCifarModel, DistillationCifarModel

SEED = 7
DATA_FOLDER = "data"


def train(experiment: str, num_workers: int = 0, is_test: bool = False,
          resume_from_checkpoint: str = None):
    seed_everything(SEED)

    hyperparams_config_function = get_test_hyperparams if is_test else get_default_hyperparams
    hyperparams_config = hyperparams_config_function(DATA_FOLDER)
    if experiment == "distillation":
        config_function = get_distillation_config
        config = config_function()
        project_name = "distillation"
        model = DistillationCifarModel(config, hyperparams_config, num_workers)
    elif experiment == "teacher":
        config_function = get_resnet_teacher_config
        config = config_function()
        project_name = f"resnet-{config.num_layers}"
        model = SingleCifarModel(config, hyperparams_config, num_workers)
    elif experiment == "student":
        config_function = get_resnet_student_config
        config = config_function()
        project_name = f"resnet-{config.num_layers}"
        model = SingleCifarModel(config, hyperparams_config, num_workers)
    else:
        raise ValueError("Unknown experiment name")

    # define logger
    wandb_logger = WandbLogger(project=project_name, log_model=True, offline=is_test)
    wandb_logger.watch(model)
    # define model checkpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath=join(wandb.run.dir, "{epoch:02d}-{val_loss:.4f}"),
        period=hyperparams_config.save_every_epoch,
        save_top_k=3,
    )

    # use gpu if it exists
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateLogger()
    trainer = Trainer(
        max_epochs=hyperparams_config.n_epochs,
        deterministic=True,
        check_val_every_n_epoch=hyperparams_config.val_every_epoch,
        row_log_interval=hyperparams_config.log_every_epoch,
        logger=wandb_logger,
        checkpoint_callback=model_checkpoint_callback,
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=gpu,
        callbacks=[lr_logger],
        reload_dataloaders_every_epoch=True,
    )

    trainer.fit(model)

    trainer.test()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("experiment", type=str, choices=["teacher", "student", "distillation"])
    arg_parser.add_argument("--n_workers", type=int, default=0)
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    args = arg_parser.parse_args()

    train(args.experiment, args.n_workers, args.test, args.resume)

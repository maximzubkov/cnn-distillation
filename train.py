from argparse import ArgumentParser
from os.path import join

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import WandbLogger

from configs import (
    get_resnet_student_default_config,
    get_resnet_student_test_config,
    get_resnet_teacher_default_config,
    get_resnet_teacher_test_config
)

from models import ResNet

SEED = 7


def train(experiment: str, num_workers: int = 0, is_test: bool = False,
          resume_from_checkpoint: str = None):
    seed_everything(SEED)

    if experiment == "train teacher":
        config_function = get_resnet_teacher_test_config if is_test else get_resnet_teacher_default_config
    elif experiment == "train student":
        config_function = get_resnet_student_test_config if is_test else get_resnet_student_default_config
    else:
        raise ValueError("Unknown experiment name")
    config = config_function()
    num_layers = config.num_layers
    model = ResNet(config, num_workers)

    # define logger
    wandb_logger = WandbLogger(project=f"resnet-{num_layers}", log_model=True, offline=is_test)
    wandb_logger.watch(model)
    # define model checkpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath=join(wandb.run.dir, "{epoch:02d}-{val_loss:.4f}"),
        period=config.hyperparams_config.save_every_epoch,
        save_top_k=3,
    )

    # use gpu if it exists
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateLogger()
    trainer = Trainer(
        max_epochs=config.hyperparams_config.n_epochs,
        deterministic=True,
        check_val_every_n_epoch=config.hyperparams_config.val_every_epoch,
        row_log_interval=config.hyperparams_config.log_every_epoch,
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
    arg_parser.add_argument("experiment", type=str, choices=["teacher", "student"])
    arg_parser.add_argument("--n_workers", type=int, default=0)
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    args = arg_parser.parse_args()

    train(args.experiment, args.n_workers, args.test, args.resume)

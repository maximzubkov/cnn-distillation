from argparse import ArgumentParser
from os import cpu_count
from os.path import join

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import WandbLogger

from utils import configure_experiment

SEED = 7


def train(experiment: str, num_workers: int = 0, is_test: bool = False,
          is_unfrozen: bool = False, resume_from_checkpoint: str = None):
    seed_everything(SEED)

    model, project_name, hyperparams_config = configure_experiment(experiment=experiment, num_workers=num_workers,
                                                                   is_test=is_test, is_unfrozen=is_unfrozen)

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
        gradient_clip_val=hyperparams_config.grad_clip
    )

    trainer.fit(model)

    trainer.test()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("experiment", type=str, choices=["teacher", "student", "kd_distillation",
                                                             "rkda_distillation", "rkdd_distillation"])
    arg_parser.add_argument("--unfrozen", action="store_true")
    arg_parser.add_argument("--n_workers", type=int, default=cpu_count())
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    args = arg_parser.parse_args()

    train(args.experiment, args.n_workers, args.test, args.unfrozen, args.resume)

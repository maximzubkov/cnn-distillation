
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything

from models import SingleCifarModel, DistillationCifarModel, LogitsDiscriminatorCifarModel

SEED = 7


def evaluate(checkpoint_path: str, experiment: str = None):
    seed_everything(SEED)
    if experiment == "distillation":
        model = DistillationCifarModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    elif (experiment == "teacher") or (experiment == "student"):
        model = SingleCifarModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    elif experiment == "discriminator":
        model = LogitsDiscriminatorCifarModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    else:
        raise ValueError("Unknown experiment name")
    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    trainer.test(model)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("experiment", type=str, choices=["teacher", "student",
                                                             "distillation", "discriminator"])

    args = arg_parser.parse_args()

    evaluate(args.checkpoint, args.experiment)

from os.path import join

from .distillation_config import DistillationConfig
from .hyperparams_config import ModelHyperparameters
from .loss_config import KDLossConfig, RKDLossConfig, LDLossConfig
from .model_config import ModelConfig


def get_default_hyperparams(data_path: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        data_path=data_path,
        n_epochs=15,
        batch_size=32,
        learning_rate=0.0003,
        weight_decay=0.0001,
        decay_gamma=0.95,
        shuffle_data=True,
    )


def get_test_hyperparams(data_path: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        data_path=data_path,
        n_epochs=10,
        batch_size=5,
        learning_rate=0.0003,
        weight_decay=0.0001,
        decay_gamma=0.95,
        shuffle_data=True,
    )


def get_kd_default_hyperparams(data_path: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        data_path=data_path,
        n_epochs=15,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.0001,
        decay_gamma=0.7,
        shuffle_data=True,
    )


def get_kd_test_hyperparams(data_path: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        data_path=data_path,
        n_epochs=10,
        batch_size=5,
        learning_rate=0.001,
        weight_decay=0.0001,
        decay_gamma=0.7,
        shuffle_data=True,
    )


def get_rkd_default_hyperparams(data_path: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        data_path=data_path,
        n_epochs=15,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.0001,
        decay_gamma=0.65,
        shuffle_data=True,
    )


def get_rkd_test_hyperparams(data_path: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        data_path=data_path,
        n_epochs=10,
        batch_size=5,
        learning_rate=0.001,
        weight_decay=0.0001,
        decay_gamma=0.65,
        shuffle_data=True,
    )


def get_resnet_teacher_config() -> ModelConfig:
    return ModelConfig(
        model_name="resnet",
        num_layers=50,
        pretrained=True,
        is_teacher=False,
        freeze_encoder=False
    )


def get_resnet_frozen_teacher_config() -> ModelConfig:
    return ModelConfig(
        model_name="resnet",
        num_layers=50,
        pretrained=True,
        is_teacher=False,
        freeze_encoder=True
    )


def get_resnet_student_config() -> ModelConfig:
    return ModelConfig(
        model_name="resnet",
        num_layers=18,
        pretrained=True,
        is_teacher=False,
        freeze_encoder=False
    )


def get_resnet_frozen_student_config() -> ModelConfig:
    return ModelConfig(
        model_name="resnet",
        num_layers=18,
        pretrained=True,
        is_teacher=False,
        freeze_encoder=True
    )


def get_kd_distillation_config() -> DistillationConfig:
    return DistillationConfig(
        teacher_config=ModelConfig(model_name="resnet",
                                   num_layers=50,
                                   pretrained=True,
                                   is_teacher=True,
                                   freeze_encoder=False,
                                   checkpoint_path=join("models", "checkpoints", "teacher_unfrozen.ckpt")),
        student_config=ModelConfig(model_name="resnet",
                                   num_layers=18,
                                   pretrained=True,
                                   is_teacher=False,
                                   freeze_encoder=False),
        loss_config=KDLossConfig(loss="KD", alpha=0.5, T=1.5)
    )


def get_frozen_kd_distillation_config() -> DistillationConfig:
    return DistillationConfig(
        teacher_config=ModelConfig(model_name="resnet",
                                   num_layers=50,
                                   pretrained=True,
                                   is_teacher=True,
                                   freeze_encoder=True,
                                   checkpoint_path=join("models", "checkpoints", "teacher.ckpt")),
        student_config=ModelConfig(model_name="resnet",
                                   num_layers=18,
                                   pretrained=True,
                                   is_teacher=False,
                                   freeze_encoder=True),
        loss_config=KDLossConfig(loss="KD", alpha=0.5, T=1.5)
    )


def get_rkdd_distillation_config() -> DistillationConfig:
    return DistillationConfig(
        teacher_config=ModelConfig(model_name="resnet",
                                   num_layers=50,
                                   pretrained=True,
                                   is_teacher=True,
                                   freeze_encoder=False,
                                   checkpoint_path=join("models", "checkpoints", "teacher_unfrozen.ckpt")),
        student_config=ModelConfig(model_name="resnet",
                                   num_layers=18,
                                   pretrained=True,
                                   is_teacher=False,
                                   freeze_encoder=False),
        loss_config=RKDLossConfig(loss="RKD_Dist", lambda_=5, T=1.5)
    )


def get_frozen_rkdd_distillation_config() -> DistillationConfig:
    return DistillationConfig(
        teacher_config=ModelConfig(model_name="resnet",
                                   num_layers=50,
                                   pretrained=True,
                                   is_teacher=True,
                                   freeze_encoder=True,
                                   checkpoint_path=join("models", "checkpoints", "teacher.ckpt")),
        student_config=ModelConfig(model_name="resnet",
                                   num_layers=18,
                                   pretrained=True,
                                   is_teacher=False,
                                   freeze_encoder=True),
        loss_config=RKDLossConfig(loss="RKD_Dist", lambda_=5, T=1.5)
    )


def get_ld_distillation_config() -> DistillationConfig:
    return DistillationConfig(
        teacher_config=ModelConfig(model_name="resnet",
                                   num_layers=50,
                                   pretrained=True,
                                   is_teacher=True,
                                   freeze_encoder=False,
                                   checkpoint_path=join("models", "checkpoints", "teacher_unfrozen.ckpt")),
        student_config=ModelConfig(model_name="resnet",
                                   num_layers=18,
                                   pretrained=True,
                                   is_teacher=False,
                                   freeze_encoder=False),
        loss_config=LDLossConfig(loss="Wasserstein", alpha=0.5, T=1.5)
    )


def get_frozen_ld_distillation_config() -> DistillationConfig:
    return DistillationConfig(
        teacher_config=ModelConfig(model_name="resnet",
                                   num_layers=50,
                                   pretrained=True,
                                   is_teacher=True,
                                   freeze_encoder=True,
                                   checkpoint_path=join("models", "checkpoints", "teacher.ckpt")),
        student_config=ModelConfig(model_name="resnet",
                                   num_layers=18,
                                   pretrained=True,
                                   is_teacher=False,
                                   freeze_encoder=True),
        loss_config=LDLossConfig(loss="Wasserstein", alpha=0.5, T=1.5)
    )

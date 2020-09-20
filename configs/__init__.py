from .distillation_config import DistillationConfig
from .hyperparams_config import ModelHyperparameters
from .model_config import ModelConfig


def get_default_hyperparams(data_path: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        data_path=data_path,
        n_epochs=100,
        batch_size=32,
        learning_rate=0.0003,
        weight_decay=0.0001,
        decay_gamma=0.95,
        shuffle_data=True,
    )


def get_test_hyperparams(data_path: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        data_path=data_path,
        n_epochs=50,
        batch_size=32,
        learning_rate=0.0003,
        weight_decay=0.0001,
        decay_gamma=0.95,
        shuffle_data=True,
    )


def get_resnet_teacher_config() -> ModelConfig:
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
        pretrained=False,
        is_teacher=False,
        freeze_encoder=False
    )


def get_distillation_config() -> DistillationConfig:
    return DistillationConfig(
        teacher_config=ModelConfig(model_name="resnet",
                                   num_layers=50,
                                   pretrained=True,
                                   is_teacher=True,
                                   freeze_encoder=True),
        student_config=ModelConfig(model_name="resnet",
                                   num_layers=18,
                                   pretrained=False,
                                   is_teacher=False,
                                   freeze_encoder=False),
    )

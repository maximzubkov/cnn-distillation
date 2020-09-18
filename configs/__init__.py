from .hyperparams_config import ModelHyperparameters
from .resnet_config import ModelConfig


def _get_default_hyperparams(data_path: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        data_path=data_path,
        n_epochs=3000,
        batch_size=512,
        learning_rate=0.01,
        weight_decay=0,
        decay_gamma=0.95,
        shuffle_data=True,
    )


def _get_test_hyperparams(data_path: str) -> ModelHyperparameters:
    return ModelHyperparameters(
        data_path=data_path,
        n_epochs=50,
        batch_size=128,
        learning_rate=0.01,
        weight_decay=0,
        decay_gamma=0.95,
        shuffle_data=True,
    )


def get_resnet_teacher_default_config(data_path: str) -> ModelConfig:
    return ModelConfig(
        num_layers=104,
        hyperparams_config=_get_default_hyperparams(data_path)
    )


def get_resnet_teacher_test_config(data_path: str) -> ModelConfig:
    return ModelConfig(
        num_layers=10,
        hyperparams_config=_get_test_hyperparams(data_path)
    )


def get_resnet_student_default_config(data_path: str) -> ModelConfig:
    return ModelConfig(
        num_layers=14,
        hyperparams_config=_get_default_hyperparams(data_path)
    )


def get_resnet_student_test_config(data_path: str) -> ModelConfig:
    return ModelConfig(
        num_layers=5,
        hyperparams_config=_get_test_hyperparams(data_path)
    )

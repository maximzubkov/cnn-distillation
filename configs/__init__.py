from .hyperparams_config import ModelHyperparameters
from .resnet_config import ResNetConfig


def _get_default_hyperparams() -> ModelHyperparameters:
    return ModelHyperparameters(
        n_epochs=3000,
        batch_size=512,
        test_batch_size=512,
        learning_rate=0.01,
        weight_decay=0,
        decay_gamma=0.95,
        shuffle_data=True,
    )


def _get_test_hyperparams() -> ModelHyperparameters:
    return ModelHyperparameters(
        n_epochs=50,
        batch_size=128,
        test_batch_size=128,
        learning_rate=0.01,
        weight_decay=0,
        decay_gamma=0.95,
        shuffle_data=True,
    )


def get_resnet_teacher_default_config() -> ResNetConfig:
    return ResNetConfig(
        num_layers=104,
        hyperparams_config=_get_default_hyperparams()
    )


def get_resnet_teacher_test_config() -> ResNetConfig:
    return ResNetConfig(
        num_layers=10,
        hyperparams_config=_get_test_hyperparams()
    )


def get_resnet_student_default_config() -> ResNetConfig:
    return ResNetConfig(
        num_layers=14,
        hyperparams_config=_get_default_hyperparams()
    )


def get_resnet_student_test_config() -> ResNetConfig:
    return ResNetConfig(
        num_layers=5,
        hyperparams_config=_get_test_hyperparams()
    )

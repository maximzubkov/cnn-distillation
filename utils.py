from configs import (
    get_resnet_student_config,
    get_resnet_frozen_student_config,
    get_resnet_teacher_config,
    get_resnet_frozen_teacher_config,
    get_test_hyperparams,
    get_default_hyperparams,
    get_kd_distillation_config,
    get_frozen_kd_distillation_config,
    get_rkda_distillation_config,
    get_frozen_rkda_distillation_config,
    get_rkdd_distillation_config,
    get_frozen_rkdd_distillation_config,
    get_kd_test_hyperparams,
    get_kd_default_hyperparams,
    get_rkd_test_hyperparams,
    get_rkd_default_hyperparams
)
from models import SingleCifarModel, DistillationCifarModel

DATA_FOLDER = "data"


def configure_experiment(experiment: str, num_workers: int = 0,
                         is_test: bool = False, is_unfrozen: bool = False):
    if experiment == "kd_distillation":
        hyperparams_config_function = get_kd_test_hyperparams if is_test else get_kd_default_hyperparams
    elif experiment == "rkd_distillation":
        hyperparams_config_function = get_rkd_test_hyperparams if is_test else get_rkd_default_hyperparams
    else:
        hyperparams_config_function = get_test_hyperparams if is_test else get_default_hyperparams
    hyperparams_config = hyperparams_config_function(DATA_FOLDER)
    freezed_flag = "unfreezed" if is_unfrozen else "freezed"
    if experiment == "kd_distillation":
        config_function = get_kd_distillation_config if is_unfrozen else get_frozen_kd_distillation_config
        config = config_function()
        project_name = f"distillation-{freezed_flag}-{config.loss_config.loss}"
        model = DistillationCifarModel(config, hyperparams_config, num_workers)
    elif experiment == "rkdd_distillation":
        config_function = get_rkdd_distillation_config if is_unfrozen else get_frozen_rkdd_distillation_config
        config = config_function()
        project_name = f"distillation-{freezed_flag}-{config.loss_config.loss}"
        model = DistillationCifarModel(config, hyperparams_config, num_workers)
    elif experiment == "rkda_distillation":
        config_function = get_rkda_distillation_config if is_unfrozen else get_frozen_rkda_distillation_config
        config = config_function()
        project_name = f"distillation-{freezed_flag}-{config.loss_config.loss}"
        model = DistillationCifarModel(config, hyperparams_config, num_workers)
    elif experiment == "teacher":
        config_function = get_resnet_teacher_config if is_unfrozen else get_resnet_frozen_teacher_config
        config = config_function()
        project_name = f"resnet-{config.num_layers}-{freezed_flag}"
        model = SingleCifarModel(config, hyperparams_config, num_workers)
    elif experiment == "student":
        config_function = get_resnet_student_config if is_unfrozen else get_resnet_frozen_student_config
        config = config_function()
        project_name = f"resnet-{config.num_layers}-{freezed_flag}"
        model = SingleCifarModel(config, hyperparams_config, num_workers)
    else:
        raise ValueError("Unknown experiment name")
    return model, project_name, hyperparams_config

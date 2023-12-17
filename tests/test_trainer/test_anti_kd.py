from pathlib import Path

from anti_kd_backdoor.config import Config
from anti_kd_backdoor.trainer import build_trainer
from anti_kd_backdoor.trainer.anti_kd import (
    AntiKDTrainer,
    NetworkWrapper,
    TriggerWrapper,
)

CONFIG_PATH = 'tests/data/config/anti_kd_t-r34_s-r18-v16-mv2_cifar10.py'


def test_anti_kd(tmp_work_dirs: Path) -> None:
    config = Config.fromfile(CONFIG_PATH)
    trainer_config = config.trainer
    trainer_config.work_dirs = tmp_work_dirs

    trainer = build_trainer(trainer_config)
    assert isinstance(trainer, AntiKDTrainer)
    assert trainer._alpha == trainer_config.alpha
    assert trainer._save_interval == trainer_config.save_interval
    assert trainer._device == trainer_config.device
    assert trainer._epochs == trainer_config.epochs

    teacher = trainer._teacher_wrapper
    assert isinstance(teacher, NetworkWrapper)
    assert teacher.lambda_t == trainer_config.teacher.lambda_t
    assert teacher.lambda_mask == trainer_config.teacher.lambda_mask
    assert teacher.trainable_when_training_trigger == \
        trainer_config.teacher.trainable_when_training_trigger

    students = trainer._student_wrappers
    for s_name, s in students.items():
        assert isinstance(s, NetworkWrapper)
        student_config = trainer_config.students
        assert s.lambda_t == getattr(student_config, s_name).lambda_t
        assert s.lambda_mask == getattr(student_config, s_name).lambda_mask
        assert s.trainable_when_training_trigger == getattr(
            student_config, s_name).trainable_when_training_trigger

    trigger = trainer._trigger_wrapper
    assert isinstance(trigger, TriggerWrapper)
    assert trigger.mask_clip_range == trainer_config.trigger.mask_clip_range
    assert trigger.trigger_clip_range == \
        trainer_config.trigger.trigger_clip_range
    assert trigger.mask_penalty_norm == \
        trainer_config.trigger.mask_penalty_norm

    clean_train_dataloader = trainer._clean_train_dataloader
    assert clean_train_dataloader.batch_size == \
        trainer_config.clean_train_dataloader.batch_size
    assert callable(clean_train_dataloader.dataset.transform)

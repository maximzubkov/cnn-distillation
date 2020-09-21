#!/bin/bash
echo "Evaluating pretrained student model with all layers frozen except BN and classifier"
python eval.py models/checkpoints/student.ckpt student
echo "Evaluating pretrained teacher model with all layers frozen except BN and classifier"
python eval.py models/checkpoints/teacher.ckpt teacher
echo "Evaluating pretrained student model with all layers unfrozen"
python eval.py models/checkpoints/student.ckpt student --unfreeze
echo "Evaluating pretrained teacher model with all layers unfrozen"
python eval.py models/checkpoints/teacher.ckpt teacher --unfreeze
echo "Evaluating pretrained student distillated by teacher, all student layers frozen except BN and classifier"
python eval.py models/checkpoints/distillation_kd.ckpt distillated
echo "Evaluating pretrained student distillated by teacher, all student layers unfrozen"
python eval.py models/checkpoints/distillation_kd_unfreezed.ckpt distillated
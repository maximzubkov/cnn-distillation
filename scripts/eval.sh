#!/bin/bash
echo "Evaluating pretrained student model with all layers frozen except BN and classifier"
python eval.py models/checkpoints/student.ckpt student
echo "Evaluating pretrained teacher model with all layers frozen except BN and classifier"
python eval.py models/checkpoints/teacher.ckpt teacher
echo "Evaluating pretrained student model with all layers unfrozen"
python eval.py models/checkpoints/student_unfrozen.ckpt student
echo "Evaluating pretrained teacher model with all layers unfrozen"
python eval.py models/checkpoints/teacher_unfrozen.ckpt teacher
echo "Evaluating pretrained student distillation by teacher, all student layers frozen except BN and classifier"
python eval.py models/checkpoints/distillation_kd.ckpt distillation
echo "Evaluating pretrained student distillation by teacher, all student layers unfrozen"
python eval.py models/checkpoints/distillation_kd_unfrozen.ckpt distillation
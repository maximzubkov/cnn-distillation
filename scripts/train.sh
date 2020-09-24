#!/bin/bash
# This script runs training for all the models with all implemented loss functions
echo "Train pretrained student model with all layers frozen except BN and classifier"
python train.py student
echo "Train pretrained student model with all layers unfrozen"
python train.py student --unfrozen

echo "Train pretrained teacher model with all layers frozen except BN and classifier"
python train.py teacher
echo "Train pretrained teacher model with all layers unfrozen"
python train.py teacher --unfrozen

echo "Train pretrained student distillated by teacher, all student layers frozen except BN and classifier"
python train.py kd_distillation
echo "Train pretrained student distillated by teacher, all student layers unfrozen"
python train.py kd_distillation --unfrozen

echo "Train pretrained student distillated by teacher using KLDiv, all student layers frozen except BN and classifier"
python train.py kd_distillation
echo "Train pretrained student distillated by teacher using KLDiv, all student layers unfrozen"
python train.py kd_distillation --unfrozen

echo "Train pretrained student distillated by teacher using RKD Distance loss"
echo "all student layers frozen except BN and classifier"
python train.py rkdd_distillation
echo "Train pretrained student distillated by teacher using RKD Distance loss"
echo "all student layers unfrozen"
python train.py rkdd_distillation --unfrozen

echo "Train pretrained student distillated by teacher using Logits Discriminator"
echo "all student layers frozen except BN and classifier"
python train.py ld_distillation
echo "Train pretrained student distillated by teacher using Logits Discriminator"
echo "all student layers unfrozen"
python train.py ld_distillation --unfrozen
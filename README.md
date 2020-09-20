# cnn-distillation
[WIP] Pytorch-lightning framework for knowledge distillation experiments with CNN

[![Github action: build](https://github.com/maximzubkov/cnn-distillation/workflows/Build/badge.svg)](https://github.com/maximzubkov/cnn-distillation/actions?query=workflow%3ABuild)

#### Results 

|          | Teacher  | Method        | Pretrained | Freeze Encoder | Accuracy |
|----------|----------|---------------|------------|----------------|----------|
| ResNet18 | ❌       | Cross Entropy |     ✅     |       ✅       |   93.1   |
| ResNet18 | ❌       | Cross Entropy |     ✅     |       ❌       |   93.6   |
| ResNet50 | ❌       | Cross Entropy |     ✅     |       ✅       |   95.7   |
| ResNet50 | ❌       | Cross Entropy |     ✅     |       ❌       |   93.8   |
| ResNet18 | ResNet50 | Embedding MSE |     ✅     |       ✅       |          |
| ResNet18 | ResNet50 | Embedding MSE |     ✅     |       ❌       |          |
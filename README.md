# cnn-distillation
[WIP] Pytorch-lightning framework for knowledge distillation experiments with CNN

[![Github action: build](https://github.com/maximzubkov/cnn-distillation/workflows/Build/badge.svg)](https://github.com/maximzubkov/cnn-distillation/actions?query=workflow%3ABuild)

#### Results 

|          | Teacher  | Method        | Pretrained | Freeze Encoder | Accuracy |
|----------|----------|---------------|------------|----------------|----------|
| ResNet18 | ❌        | Cross Entropy |     ✅     |       ✅       |   93.07  |
| ResNet18 | ❌        | Cross Entropy |     ✅     |       ❌       |   93.65   |
| ResNet50 | ❌        | Cross Entropy |     ✅     |       ✅       |   95.71   |
| ResNet50 | ❌        | Cross Entropy |     ✅     |       ❌       |   93.83   |
| ResNet18 | ResNet50  | Embedding MSE |     ✅     |       ✅       |  93.29   |
| ResNet18 | ResNet50  | Embedding MSE |     ✅     |       ❌       |   94.2   |
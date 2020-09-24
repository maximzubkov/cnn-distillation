# cnn-distillation
[WIP] Pytorch-lightning framework for knowledge distillation experiments with CNN

[![Github action: build](https://github.com/maximzubkov/cnn-distillation/workflows/Build/badge.svg)](https://github.com/maximzubkov/cnn-distillation/actions?query=workflow%3ABuild)

-------
#### Введение

Данный репозиторий содержит в себе решение вступительного испытание в VK lab.
Задание заключается в том, чтобы придумать и поставить эксперементы показывающие 
работоспособность метода `Knowledge Distillation`. Перед тем как приступить к 
решению задачи, я прежде всего решил написать хороший и читаемый репозиторий, а не 
серию ноутбуков. Мне это кажется это важным, так как на мой взгляд в VK многие 
разработки имеют шанс быть использованными в продакшене. Для достижения этих целей я
использовал популярный фреймворки [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/l…) 
и [W&B](https://wandb.ai) для визуализации. Также я настроил базовый `CI` пайплайн 
с помощью `Github Actions`, а большие файлы с `checkpoint` моделей
я хранил при помощи `github-lfs`. 

-------
#### Постановка задачи и шаги в решении

Цель данной рабобы заключается в том, чтобы изучить некоторые существующие подходы 
`Knowledge Distillation`, а также реализовать фрейморк, который можно 
было бы в последствии расширить. В качетсве датасета я использовал датасет `cifar10` 
из-за его простоты,  а также того факта, что он есть в `torchvision`, а все эксперемнты
были сделаны над моедлями `Resnet`.

Работу можно разделить на несколько этапов:
> Для начала я решил реализовать классы `BaseCifarModel` и `SingleCifarModel` 
> базовые модели `Resnet18` и `Resnet50`, обучить их на `cifar10`. 
> Каждую модель я обучил как `from scratch`, так и c заморозкой нескольких слоев. 
> Первый мой эксперемнт был в том, что я в `Resnet` заморозил все слои кроме классификатора, 
> но полученные результаты были низкие: `accuracy` порядка 85%. Далее я наткнулся на 
> статью на [kaggle](https://www.kaggle.com/nkaenzig/cnn-transfer-learning-secrets) и 
> воспользовался оттуда двумя советами:
> * Увеличить размер входной картинки (поэксперементировав я остановился на `128x128`). Так как исходная модель была обучена на
> `ImageNet` (размер картинок в котором `224x224`), то такой размер будет для модели более приятным
> * Разморозить `BatchNormalization`. Данное улучшение также кажется логичным, так как свертки обучены на `ImageNet`
> достаточно хороши, а вот масштабы и локализация объектов в `cifar10` может немного отличать от `ImageNet`

> Далее я реализовал `DistillationCifarModel` и `loss` функцию из следующей 
> [статьи](http://cs230.stanford.edu/files_winter_2018/projects/6940224.pdf), 
> результаты получились хорошими не сразу достаточно много времени пришлось 
> потратить на подбор гиперпарамтеров

> Я имею опыт работы с `GAN` и поэтому мне показалось хоршей идея использовать
> своего рода дискриминатор для `logits` моделей. Дискриминатор принимает в себя 
> `logits` студента и выдает вероятность того, что данные `logits` получены от 
> учителя. При этом дискриминатор и студент учатся по очереди, я тестировал разные 
> стартегии, к примеру "30 шаго учится дискриминатор,
> а следующие 170 студент". Для реализации этого я использовал. К сожалению
> экспреемнты которые я поставил не увенчались успехом, я заметил, что результат на 
> валидации тем лучше, чем меньше шагов учится дискриминатор. Я 

> Наконец, не получив ожидаемого результата с `GAN` я решил воспроизвести результат
> статьи [RKD](https://arxiv.org/pdf/1904.05068.pdf), к счастью тут я поучил более 
> позитивные результаты

> Затем мне стало инетерсно сравнить куда смотрят те или ины сети, когда делают свои 
> предсказани и насколько сильно "взгляд" студента похож на взгляд учителя при 
> использовании `Knowledge Distillation`. Для этого я воспользовался подходом 
> [grad-cam](https://arxiv.org/abs/1610.02391) реализованный в библиотеке `gradcam`

#### Запуск экспрементов

> Перед запуском эксперементов необходимо установить все неоходимые библиотеки, 
это можно сделать при помощи команды
> ```bash
> sh scripts/build.sh
> ```

> Все эксперемнты можно запустить используя команду 
> ```bash
> sh scripts/train.sh
> ```
> Если необходимо запустить обучения какого-то конкретного экспреемента необходимо 
> выполнить команду
> ```bash
> python train.py <experiment name> 
> ```
> Таже можно добавить флаг `--unfrozen` чтобы обучить модель со всеми слоями размороженными 
> Названия эксперемнтов следующие:
> * Обучение студента без учителя: `student`
> * Обучение учителя: `teacher`
> * Обучение студента c учителем с использованием KD Loss 
> из [статьи](http://cs230.stanford.edu/files_winter_2018/projects/6940224.pdf): `kd_distillation`
> * Обучение студента c учителем с использованием RKD Distance Loss 
> из [статьи](https://arxiv.org/pdf/1904.05068.pdf): `rkdd_distillation`
> * Обучение студента c учителем с использованием Logits Discriminator Loss 
> из [статьи](https://arxiv.org/pdf/1904.05068.pdf): `ld_distillation`

> Обученные модели сохранены в папке `models/checkpoints`. Чтобы получить оценки 
> качества всех моделей необходимо выполнить команду 
> ```bash
> sh scripts/eval.sh
> ```
> Если необходимо метрики какого-то конкретного экспреемента необходимо 
> выполнить команду
>```bash
> python eval.py <path to .ckpt file> <experiment name>
> ```
> В данном случае видов эксперементов всего три: [`teacher`,`student`,`distillation`],
> а соответсвующие эксперементам `.ckpt` названия файлов представлены в таблице с результатами 

-------
#### Результаты 

| Student  | Teacher  | Method                | Pretrained | Freeze Encoder | Accuracy | `.ckpt` file                  |
|----------|----------|-----------------------|------------|----------------|----------|-------------------------------|
| ResNet18 | ❌        | Cross Entropy        |     ✅      |       ✅       |  93.07   |student.ckpt                   |
| ResNet18 | ❌        | Cross Entropy        |     ✅      |       ❌       |  93.65   |student_unfrozen.ckpt          | 
| ResNet50 | ❌        | Cross Entropy        |     ✅      |       ✅       |  95.71   |teacher.ckpt                   |
| ResNet50 | ❌        | Cross Entropy        |     ✅      |       ❌       |  93.83   |teacher_unfrozen.ckpt          |
| ResNet18 | ResNet50  | Default KD loss      |     ✅     |       ✅        |  93.29   |distillation_kd.ckpt           |
| ResNet18 | ResNet50  | Default KD loss      |     ✅     |       ❌        |  94.26   |distillation_kd_unfrozen.ckpt  |
| ResNet18 | ResNet50  | RKD Distance loss    |     ✅     |       ✅        |  93.24   |distillation_rkdd.ckpt         |
| ResNet18 | ResNet50  | RKD Distance loss    |     ✅     |       ❌        |  94.43   |distillation_rkdd_unfrozen.ckpt|
| ResNet18 | ResNet50  | Logits Discriminator |     ✅     |       ✅        |     |    |
| ResNet18 | ResNet50  | Logits Discriminator |     ✅     |       ❌        |     |    |

#### 
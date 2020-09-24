
from argparse import ArgumentParser

import torchvision.transforms as transforms
from gradcam import GradCAMpp
from gradcam.utils import visualize_cam
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid

from models import SingleCifarModel, DistillationCifarModel

SEED = 2
DATA = "data"


def evaluate(distilled_checkpoint_path: str, teacher_checkpoint_path: str, student_checkpoint_path: str):
    seed_everything(SEED)
    models = {
        "distillated": DistillationCifarModel.load_from_checkpoint(checkpoint_path=distilled_checkpoint_path).student,
        "teacher": SingleCifarModel.load_from_checkpoint(checkpoint_path=teacher_checkpoint_path).model,
        "student": SingleCifarModel.load_from_checkpoint(checkpoint_path=student_checkpoint_path).model
    }
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor()])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = CIFAR10(root=DATA,
                      train=False,
                      download=True,
                      transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    image, labels = next(iter(dataloader))
    plot = [image.cpu().squeeze()]
    for _, model in models.items():
        model.eval()
        cam = GradCAMpp(arch=model, target_layer=model.encoder.layer4)
        mask, _ = cam(normalize(image.squeeze()).unsqueeze(0))
        _, result = visualize_cam(mask, image)
        plot.append(result)

    grid_image = make_grid(plot, nrow=1)
    transforms.ToPILImage()(grid_image).show()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("distilled_checkpoint", type=str)
    arg_parser.add_argument("teacher_checkpoint", type=str)
    arg_parser.add_argument("student_checkpoint", type=str)

    args = arg_parser.parse_args()

    evaluate(args.distilled_checkpoint, args.teacher_checkpoint, args.student_checkpoint)

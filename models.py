# models.py

import torch
import torch.nn.functional as F
from torchvision import models
import os
import requests
from robustness import model_utils
from robustness.datasets import ImageNet

# Mean and std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mu = torch.tensor(mean).view(3, 1, 1).cuda()
std = torch.tensor(std).view(3, 1, 1).cuda()

# URLs for robust models
robust_urls = {
    "resnet18_linf_eps0.5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet18_linf_eps0.5.ckpt",
    "resnet18_linf_eps1.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet18_linf_eps1.0.ckpt",
    "resnet18_linf_eps2.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet18_linf_eps2.0.ckpt",
    "resnet18_linf_eps4.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet18_linf_eps4.0.ckpt",
    "resnet18_linf_eps8.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet18_linf_eps8.0.ckpt",
    "resnet50_linf_eps0.5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps0.5.ckpt",
    "resnet50_linf_eps1.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps1.0.ckpt",
    "resnet50_linf_eps2.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps2.0.ckpt",
    "resnet50_linf_eps4.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps4.0.ckpt",
    "resnet50_linf_eps8.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps8.0.ckpt",
    "wide_resnet50_2_linf_eps0.5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_linf_eps0.5.ckpt",
    "wide_resnet50_2_linf_eps1.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_linf_eps1.0.ckpt",
    "wide_resnet50_2_linf_eps2.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_linf_eps2.0.ckpt",
    "wide_resnet50_2_linf_eps4.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_linf_eps4.0.ckpt",
    "wide_resnet50_2_linf_eps8.0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_linf_eps8.0.ckpt",
}

class Model(torch.nn.Module):
    def __init__(self, backbone="18", path="./pretrained_models/"):
        super().__init__()
        self.norm = lambda x: (x - mu) / std
        if backbone == "152":
            self.backbone = models.resnet152(pretrained=True)
        elif backbone == "50":
            self.backbone = models.resnet50(pretrained=True)
        elif backbone == "18":
            self.backbone = models.resnet18(pretrained=True)
        else:
            self.backbone = RobustModel(arch=backbone, path=path).model

        self.backbone.fc = torch.nn.Identity()
        
    def forward(self, x):
        x = self.norm(x)
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

class RobustModel(torch.nn.Module):
    def __init__(self, arch="resnet50_linf_eps2.0", path="./pretrained_models/"):
        super().__init__()
        ckpt_path = download_and_load_backbone(robust_urls[arch], arch, path)
        self.model, _ = resume_finetuning_from_checkpoint(
            ckpt_path, "_".join(arch.split("_")[:-2])
        )
        self.model = self.model.model

    def forward(self, x):
        return self.model(x)

def resume_finetuning_from_checkpoint(finetuned_model_path, arch):
    """Given arguments, dataset object and a finetuned model_path, returns a model
    with loaded weights and returns the checkpoint necessary for resuming training.
    """
    print("[Resuming finetuning from a checkpoint...]")
    model, checkpoint = model_utils.make_and_restore_model(
        arch=arch, dataset=ImageNet("/imagenet/"), resume_path=finetuned_model_path
    )
    return model, checkpoint

def download_and_load_backbone(url, model_name, path):
    arch = "_".join(model_name.split("_")[:-2])
    print(f"{arch}, {model_name}")
    os.makedirs(path, exist_ok=True)
    ckpt_path = os.path.join(path, f"{model_name}.ckpt")

    # Check if checkpoint file already exists
    if os.path.exists(ckpt_path):
        print(f"{model_name} checkpoint file already exists.")
        return ckpt_path

    r = requests.get(url, allow_redirects=True)  # to get content after redirection
    with open(ckpt_path, "wb") as f:
        f.write(r.content)

    return ckpt_path
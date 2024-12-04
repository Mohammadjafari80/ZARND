# main.py

import torch
import argparse
from tqdm import tqdm
import os

# Import from our modules
from models import Model
from datasets import get_loaders
from utils import get_score, get_adv_score

global Logger
Logger = None

def log(msg):
    global Logger
    Logger.write(f"{msg}\n")
    print(msg)

def main(args):
    log(
        "Dataset: {}, Normal Label: {}".format(
            args.source_dataset, args.label
        )
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(device)
    model = Model(str(args.backbone), args.model_path)
    model = model.to(device)
    model.eval()
    
    train_loader, test_loader  = get_loaders(
        source_dataset=args.source_dataset,
        target_datset=args.target_dataset,
        label_class=args.label,
        batch_size=args.batch_size,
        backbone=args.backbone,
        source_path=args.source_dataset_path,
        target_path=args.target_dataset_path,
        test_type=args.test_type,
    )

    # Calculate the scores immediately after loading the model and data loaders
    auc = get_score(model, device, train_loader, test_loader)
    log(f"Test AUC score: {auc}")

    # If adversarial attacks are specified, compute adversarial scores
    if args.test_attacks:
        eps = eval(args.eps)
        for attack_type in args.test_attacks:
            adv_auc = get_adv_score(
                model, device, train_loader, test_loader, attack_type, eps
            )
            log(f"Attack Type: {attack_type}, Adv AUC: {adv_auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("--source_dataset", default="cifar10")
    parser.add_argument("--source_dataset_path", default="~/cifar10", type=str)
    parser.add_argument("--target_dataset")
    parser.add_argument("--target_dataset_path", default="~/cifar100", type=str)
    
    parser.add_argument("--model_path", default="./pretrained_models/", type=str)
    
    parser.add_argument("--label", type=int, help="The normal class")
    
    parser.add_argument("--eps", type=str, default="2/255", help="The esp for attack.")
    parser.add_argument(
        "--test_type",
        type=str,
        default="ad",
        choices=["ad", "osr", "ood"],
        help="the type of test",
    )
    
    parser.add_argument("--batch_size", default=128, type=int)
    
    parser.add_argument(
        "--backbone",
        choices=[
            "resnet18_linf_eps0.5",
            "resnet18_linf_eps1.0",
            "resnet18_linf_eps2.0",
            "resnet18_linf_eps4.0",
            "resnet18_linf_eps8.0",
            "resnet50_linf_eps0.5",
            "resnet50_linf_eps1.0",
            "resnet50_linf_eps2.0",
            "resnet50_linf_eps4.0",
            "resnet50_linf_eps8.0",
            "wide_resnet50_2_linf_eps0.5",
            "wide_resnet50_2_linf_eps1.0",
            "wide_resnet50_2_linf_eps2.0",
            "wide_resnet50_2_linf_eps4.0",
            "wide_resnet50_2_linf_eps8.0",
            "18",
            "50",
            "152",
        ],
        default="18",
        type=str,
        help="ResNet Backbone",
    )

    parser.add_argument(
        "--test_attacks", help="Desired Attacks for adversarial test", nargs="+"
    )
    
    args = parser.parse_args()

    if args.test_type == "ood":
        assert args.label is None
        assert args.target_dataset is not None
    elif args.test_type == "osr":
        assert args.label is None
        assert args.target_dataset is None
    else:
        assert args.label is not None
        assert args.target_dataset is None

    os.makedirs(f"./results/{args.test_type}/", exist_ok=True)

    file_name = f"ZARND-{args.source_dataset}-{args.label}-ResNet{args.backbone}-eps-{eval(args.eps)}-{args.test_type}.txt"
    file_path = f"./results/{args.test_type}/{file_name}"

    # Check if the file already exists
    if os.path.exists(file_path):
        # If it does, find a new file name by appending a number to the end
        i = 1
        while os.path.exists(f"./results/{args.test_type}/{file_name[:-4]}_{i}.txt"):
            i += 1
        file_name = f"{file_name[:-4]}_{i}.txt"

    Logger = open(f"./results/{args.test_type}/{file_name}", "a", encoding="utf-8")

    main(args)

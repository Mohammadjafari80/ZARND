# utils.py

import torch
import numpy as np
import faiss
from sklearn.metrics import roc_auc_score
import gc
from tqdm import tqdm

# Import attack modules
from KNN import KnnFGSM, KnnPGD, KnnAdvancedPGD

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for imgs, _ in tqdm(train_loader, desc="Train set feature extracting"):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = (
            torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
        )
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Test set feature extracting"):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = (
            torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        )
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc


def get_adv_score(model, device, train_loader, test_loader, attack_type, eps):
    train_feature_space = []
    with torch.no_grad():
        for imgs, _ in tqdm(train_loader, desc="Train set feature extracting"):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features.detach().cpu())
        train_feature_space = (
            torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
        )

    mean_train = torch.mean(torch.Tensor(train_feature_space), axis=0)

    gc.collect()
    torch.cuda.empty_cache()

    test_attack = None
    if attack_type.startswith("PGDA"):
        steps = int(attack_type.split("-")[1])
        test_attack = KnnAdvancedPGD.PGD_KNN_ADVANCED(
            model,
            train_feature_space,
            eps=eps,
            steps=steps,
            alpha=(2.5 * eps) / steps,
            k=2,
        )
    elif attack_type.startswith("PGD"):
        steps = int(attack_type.split("-")[1])
        test_attack = KnnPGD.PGD_KNN(
            model,
            mean_train.to(device),
            eps=eps,
            steps=steps,
            alpha=(2.5 * eps) / steps,
        )
    else:
        test_attack = KnnFGSM.FGSM_KNN(model, mean_train.to(device), eps=eps)

    test_adversarial_feature_space = []
    adv_test_labels = []

    for imgs, labels in tqdm(
        test_loader, desc="Test set adversarial feature extracting"
    ):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        adv_imgs, adv_imgs_in, adv_imgs_out, labels = test_attack(imgs, labels)
         
        adv_test_labels += labels.cpu().numpy().tolist()
        del imgs, labels

        adv_features = model(adv_imgs)
        test_adversarial_feature_space.append(adv_features.detach().cpu())
        del adv_features, adv_imgs
        
        torch.cuda.empty_cache()

    test_adversarial_feature_space = (
        torch.cat(test_adversarial_feature_space, dim=0)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )

    adv_distances = knn_score(train_feature_space, test_adversarial_feature_space)
    adv_auc = roc_auc_score(adv_test_labels, adv_distances)

    del (
        test_adversarial_feature_space,
        adv_distances,
        adv_test_labels,
    )
    
    gc.collect()
    torch.cuda.empty_cache()

    return adv_auc
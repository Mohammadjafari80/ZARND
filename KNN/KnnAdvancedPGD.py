from KNN.KnnAttack import Attack
import torch
import torch.nn as nn

class PGD_KNN_ADVANCED(Attack):
    def __init__(self, model, train_embeddings, k=2, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD_KNN_ADVANCED", model)
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.random_start = random_start
        self.k = k
        self.device = next(model.parameters()).device
        self.train_embeddings = torch.tensor(train_embeddings, device=self.device)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Separate normal and anomaly images based on labels
        normal_images = images[labels == 0]
        anomaly_images = images[labels == 1]

        def adv_attack(target_images, attack_anomaly):
            adv_images = target_images.clone().detach()

            if adv_images.numel() == 0:
                return adv_images

            if self.random_start:
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for _ in range(self.steps):
                adv_images.requires_grad = True

                outputs = self.model(adv_images)  # Shape: (batch_size, feature_dim)

                # Compute distances between outputs and train_embeddings
                distances = torch.cdist(outputs, self.train_embeddings)  # Shape: (batch_size, num_train_samples)

                # Find the k nearest neighbors
                knn_distances, _ = distances.topk(self.k, largest=False, dim=1)  # Shape: (batch_size, k)

                # Compute cost as mean of the distances to the K-nearest neighbors
                cost = knn_distances.mean()

                if attack_anomaly:
                    cost = -cost

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - target_images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(target_images + delta, min=0, max=1).detach()

            return adv_images

        # Perform attacks on normal and anomaly images
        adv_normal_images = adv_attack(normal_images, attack_anomaly=False)
        adv_anomaly_images = adv_attack(anomaly_images, attack_anomaly=True)

        # Combine the adversarial images
        adv_images = torch.cat([adv_normal_images, adv_anomaly_images], dim=0)
        adv_images_in = torch.cat([adv_normal_images, anomaly_images], dim=0)
        adv_images_out = torch.cat([normal_images, adv_anomaly_images], dim=0)

        # Create the targets: 0 for normal samples, 1 for anomalies
        targets_normal = torch.zeros(adv_normal_images.size(0), device=self.device)
        targets_anomaly = torch.ones(adv_anomaly_images.size(0), device=self.device)
        targets = torch.cat([targets_normal, targets_anomaly], dim=0)

        return adv_images, adv_images_in, adv_images_out, targets

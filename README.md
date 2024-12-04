### Killing It With Zero-Shot: Adversarially Robust Novelty Detection

---

#### Overview
This repository contains the official implementation of the paper **"Killing It With Zero-Shot: Adversarially Robust Novelty Detection"**, presented at the ICASSP 2024 conference. The paper introduces a novel approach to novelty detection (ND) by leveraging adversarial robustness and robust features extracted from pre-trained models. Our method significantly improves the performance of ND under adversarial conditions, bridging the gap between nearest-neighbor techniques and robust feature-based methods.

---

#### Features
- **Zero-Shot Novelty Detection**: Leverages pre-trained models for robust anomaly detection without task-specific fine-tuning.
- **Adversarial Robustness**: Provides resistance against adversarial attacks such as PGD and FGSM.
- **Multi-Testing Modes**:
  - **Anomaly Detection (AD)**
  - **Open Set Recognition (OSR)**
  - **Out-of-Distribution Detection (OOD)**
- **Configurable Testing**: Customizable adversarial attacks, backbone models, datasets, and parameters.

---

#### Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mohammadjafari80/ZARND.git
   cd ZARND
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare datasets**:
   Ensure datasets are downloaded and paths are updated accordingly (e.g., `~/cifar10` for CIFAR-10).

4. **Run the code**:
   ```bash
   python main.py --source_dataset cifar10 --label 0 --backbone resnet18_linf_eps8.0 --test_type ad --test_attacks PGD-10 --eps 4/255
   ```

---

#### Arguments

| Argument               | Default Value           | Description                                                                                                                                                     |
|------------------------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--source_dataset`     | `cifar10`              | Source dataset for training and testing.                                                                                                                        |
| `--source_dataset_path`| `~/cifar10`            | Path to the source dataset.                                                                                                                                    |
| `--target_dataset`     | None                   | Target dataset (only used for OOD testing).                                                                                                                    |
| `--target_dataset_path`| `~/cifar100`           | Path to the target dataset (only used for OOD testing).                                                                                                        |
| `--model_path`         | `./pretrained_models/` | Path to pre-trained models.                                                                                                                                    |
| `--label`              | None                   | The class to consider as "normal" for anomaly detection. Must be specified for AD tests.                                                                       |
| `--eps`                | `4/255`                | Perturbation limit for adversarial attacks.                                                                                                                    |
| `--test_type`          | `ad`                   | Type of test to perform. Options: `ad` (Anomaly Detection), `osr` (Open Set Recognition), `ood` (Out-of-Distribution Detection).                                |
| `--batch_size`         | 128                    | Batch size for data loading.                                                                                                                                   |
| `--backbone`           | `18`                   | Backbone model to use. Options include ResNet variants with different robustness (e.g., `resnet18_linf_eps2.0`, `wide_resnet50_2_linf_eps4.0`) or standard ResNets (`18`, `50`). |
| `--test_attacks`       | None                   | List of adversarial attacks to test. Options: `PGD-n` (PGD with `n` steps), `PGDA-n` (advanced PGD with `n` steps), or `FGSM`.                                  |

---

#### Adversarial Testing

The adversarial tests are configurable using the `--test_attacks` argument. Supported attacks:
- **FGSM**: Fast Gradient Sign Method.
- **PGD-n**: Projected Gradient Descent with `n` steps.
- **PGDA-n**: Advanced PGD with `n` steps.

For example, to test with FGSM and PGD with 10 steps:
```bash
python main.py --source_dataset cifar10 --label 0 --test_attacks FGSM PGD-10
```

---

#### Results
Results are saved in the `./results/<test_type>/` directory. Filenames include dataset, backbone, and test type for easy identification.

---

#### Citation
If you find this code useful, please cite our work:
```
@INPROCEEDINGS{10446155,
  author={Mirzaei, Hossein and Jafari, Mohammad and Dehbashi, Hamid Reza and Sadat Taghavi, Zeinab and Sabokrou, Mohammad and Rohban, Mohammad Hossein},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Killing It With Zero-Shot: Adversarially Robust Novelty Detection}, 
  year={2024},
  pages={7415-7419},
  keywords={anomaly detection; adversarial robustness; zero-shot learning},
  doi={10.1109/ICASSP48485.2024.10446155}
}
```

---

Feel free to raise an issue or contribute to this repository!
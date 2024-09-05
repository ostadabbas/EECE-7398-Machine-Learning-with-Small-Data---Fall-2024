# EECE 7398: Machine Learning with Small Data - Fall 2024

## Leveraging the Discovery Cluster for Advanced ML Research

### Instructor: Prof. Sarah Ostadabbas
#### Northeastern University, Department of Electrical and Computer Engineering

---

## Overview

This repository contains the materials for the **EECE 7398: Machine Learning with Small Data** course. The course emphasizes using Northeastern's Discovery Cluster for advanced machine learning experiments, particularly in small data scenarios.

The materials include pre-prepared slides (uploaded directly as `.pdf`) and hands-on exercises to guide students through accessing, setting up, and utilizing the Discovery Cluster for machine learning research. Topics covered:

- Introduction to the Discovery Cluster
- Cluster access and environment configuration
- PyTorch setup and GPU utilization
- Monitoring experiments with Weights & Biases
- Best practices for managing machine learning experiments

---

## Contents

- `Slides/`: Uploaded PDF slides covering session topics
- `cifar10_classification_assignment_corrected.ipynb`: CIFAR-10 classification assignment with W&B monitoring
- `cifar10_training_inference_part3.ipynb`: Notebook for training/inference on the Discovery Cluster
- `wandb_monitoring_script.py`: Script for setting up W&B monitoring
- `conda_environment_setup_part1.ipynb`: Notebook for setting up the Conda environment
- `README.md`: This readme file

---

## Prerequisites

Before using the Discovery Cluster for this course, students must have:

1. **Access to the Discovery Cluster**: [Request access through ServiceNow](https://rc.northeastern.edu/getting-started/)
2. Basic knowledge of **Python** and **machine learning**
3. Familiarity with **Conda** for environment management

---

## Exercises Overview

1. **CIFAR-10 Classification and W&B Monitoring** (`cifar10_classification_assignment_corrected.ipynb`)
   - Train and track a custom CNN on the CIFAR-10 dataset.
   - Utilize Weights & Biases (W&B) to monitor training performance.
   
2. **Training and Inference on GPUs** (`cifar10_training_inference_part3.ipynb`)
   - Train a simple classifier using the CIFAR-10 dataset on the Discovery Cluster GPUs.
   - Run inference on pre-trained models.
   
3. **W&B Monitoring Setup** (`wandb_monitoring_script.py`)
   - A Python script to set up W&B for experiment tracking without Jupyter Notebooks.
   
4. **Setting up Conda Environment** (`conda_environment_setup_part1.ipynb`)
   - Step-by-step guide to creating a Conda environment with the required dependencies for GPU-accelerated PyTorch on the Discovery Cluster.

---

## How to Use

1. Clone the repository:

    ```
    git clone https://github.com/your-repo/EECE7398-ML-Small-Data-Fall2024.git
    ```

2. Review the provided materials:
    - Use the uploaded slides in `Slides/` to follow along with the course sessions.
    - Work through the exercises in the provided Jupyter notebooks.

3. Access the cluster and follow the steps outlined in the slides to set up your environment for machine learning experimentation.

---

## Additional Resources

- [RC Discovery Cluster Documentation](https://rc.northeastern.edu)
- [Conda Documentation](https://docs.conda.io/projects/conda/en/latest/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Weights and Biases Documentation](https://docs.wandb.ai/)

For any issues or questions, feel free to reach out to the Northeastern RC support team via [ServiceNow](https://rc.northeastern.edu/help/support/).

---

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

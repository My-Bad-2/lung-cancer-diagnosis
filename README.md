# Lung Cancer Diagnosis
This repository contains the Python implementation of a sophisticated, three-stage deep learning pipeline for the classification of lung cancer. The project leverages three distinct data modalities—histopathological images, CT scans, and tabular clinical data—by training a specialized, state-of-the-art neural network for each.

The pipeline is designed to be modular, allowing for independent training and evaluation of each component model.

## Model Architecture
The core of this project lies in its use of three distinct, advanced neural network architectures, each tailored to its specific data type.

### Stage 1: Histopathological Subtype classification (`HistoNet`)
This stage is designed to analyzes high-resolution tissue slide images to classify cancer subtypes.

- **Architecture**: A custom-built Convolutional Neural Network (CNN).
- **Key Features**:
  - **Inception Modules**: Employs parallel convolutional filters of different sizes (1x1, 3x3, 5x5) to capture features at multiple scales simultaneously, which is ideal for the varied structures in histopathology.
  - **Residual Blocks**: Incorporates skip connections (from ResNet) to enable the training of a much deeper network without suffering from vanishing gradients, leading to more robust feature extraction.
  - **GELU Activation**: Uses a modern, smooth activation function for improved performance.

### Stage 2: Computed Tomography Scan analysis (`CTS_CapsNet`)
This model analyzes CT scan images to identify malignancies.

- **Architecture**: A Capsule Network (CapsNet).
- **Key Features**:
  - **Spatial Hierarchy**: Unlike traditional CNNs which can lose spatial information through pooling layers, CapsNets preserve the hierarchical relationships between features. This is critical in medical imaging for understanding the orientation and relative position of anatomical structures.
  - **Routing-by-Agreement**: An internal mechanism that allows the network to recognize whole objects by composing parts, making it more robust to changes in viewpoint and rotation.
  - **CapsuleLoss**: Utilizes a specialized margin loss function tailored for capsule outputs.

### Stage 3: Clinical Data Analysis (`Clinical_DBN`)
This model is designed to find complex, non-linear patterns within tabular patient data (e.g., age, smoking history, genetic markers).

- **Architecture**: A Deep Belief Network (DBN).
- **Key Features**:
  - **Generative Pre-training**: The DBN is composed of stacked Restricted Boltzmann Machines (RBMs). It undergoes a unique two-phase training process:
    1. **Unsupervised Pre-training**: Each RBM layer is trained greedily, one after another, to learn the underlying probability distribution of the input data without using labels. This initializes the network weights in a highly effective region of the parameter space. 
    2. **Supervised Fine-tuning**: After pre-training, a classifier is added, and the entire network is fine-tuned using labels to optimize for the specific prediction task.

## Dataset Structure
To use this pipeline, the data must be organized into the following folder structure. The `main()` function in the script expects these paths.
```
/your_main_data_directory/
├── histopathological_images/
│   ├── class_adenocarcinoma/
│   │   ├── image_001.png
│   │   └── ...
│   ├── class_squamous_cell/
│   │   ├── image_101.png
│   │   └── ...
│   └── ...
├── ct_scans/
│   ├── class_benign/
│   │   ├── slice_01.png
│   │   └── ...
│   ├── class_adenocarcinoma/
│   │   └── ...
│   └── ...
└── clinical_data.csv
```

**Note**: The clinical_data.csv file should contain features in the initial columns and the target label in the final column.

## Running
### Prerequisites
1. Python 3.9+
2. PyTorch
3. TorchVision
4. Pandas
5. Numpy
6. An NVIDIA GPU is highly recommended for reasonable training times.

### Installation
1. Clone this repository:
```shell
git clone [https://github.com/My-Bad-2/lung-cancer-diagnosis](https://github.com/My-Bad-2/lung-cancer-diagnosis)
cd lung-cancer-diagnosis
```
2. Install the required packages:
```shell
python -m pip install -r prerequisites.txt
```
3. Confiuration
Open the `main.py` script and update the path variables in the configuration dictionary to point to your dataset locations.
4. Execution
Run the main script from your terminal. The pipeline will proceed to train each of the three models sequentially. The best weights for each model will be saved to a .pt file in the root directory.
```shell
python src/main.py
```

The script includes modern training utilities such as:
1. AdamW Optimizer with L1 and L2 Regularization.
2. Cosine Annealing Learning Rate Scheduler to help find better minima.
3. Early Stopping to prevent overfitting and save the best model automatically.

## License
This project is licensed under the MIT License. See the `[LICENSE](LICENSE.md)` for more details.
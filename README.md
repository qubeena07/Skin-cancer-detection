# Skin Cancer Detection Using CNN

This project implements a Convolutional Neural Network (CNN) to classify skin lesions as either malignant or benign using a dataset of labeled images. It leverages TensorFlow and Keras for building and training the model.

Find the code in Kaggle - (https://www.kaggle.com/code/dipikaranabhat/skin-cancer-detection)
---

## Dataset
The dataset used for this project is available on Kaggle: [Skin Cancer: Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign/code). 

### Dataset Structure:
- **Train Directory**: Contains the training images.
- **Test Directory**: Contains the validation/test images.

Download and place the dataset in your project directory under the folder `archive/`.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- Virtual environment (optional but recommended)

### Steps to Install Dependencies
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the dataset is in the `archive/` directory with subfolders for `train` and `test`.

---

## Running the Project

1. **Set Up Dataset Paths**:
   Update the paths in the notebook if needed:
   ```python
   train_path = '/path/to/archive/train'
   val_path = '/path/to/archive/test'
   ```

2. **Run the Notebook**:
   Open and execute the Jupyter notebook:
   ```bash
   jupyter notebook Skin-Classification.ipynb
   ```

3. **Training the Model**:
   The notebook includes code to preprocess the data, define the CNN architecture, and train the model. The training dataset contains 2,637 images, and the validation dataset contains 660 images.

4. **Evaluation**:
   After training, the model is evaluated on the test dataset for accuracy and loss.

---

## Key Components
- **Image Preprocessing**: Images are resized to 224x224 pixels and batched for training.
- **CNN Model**: The architecture uses layers from Keras for feature extraction.
- **Metrics**: Accuracy and loss are tracked during training and evaluation.

---

## Example Outputs
Add any visualizations or metrics (e.g., training/validation accuracy, confusion matrix) here.

---

## References
- Dataset: [Skin Cancer: Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign/code)
- Frameworks: TensorFlow, Keras

---

## License
Specify the license under which this code is shared, such as MIT or GPL.

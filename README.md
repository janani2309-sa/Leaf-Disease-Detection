# Plant Disease Classification with Convolutional Neural Networks

## Overview

This project demonstrates the use of Convolutional Neural Networks (CNNs) for classifying plant images to identify diseases. We utilize TensorFlow and Keras to build, train, and evaluate a CNN model that predicts whether a plant is suffering from a disease based on images of its leaves.

## Dataset

The dataset used in this project is sourced from a publicly available plant image dataset. It includes images of three different plant diseases:

- **Corn-Common Rust**
- **Potato-Early Blight**
- **Tomato-Bacterial Spot**

The dataset is organized into directories corresponding to each disease category, containing multiple images per category. The images are pre-processed to a consistent size of 256x256 pixels.

## Techniques and Models Used

### 1. Data Preprocessing

- **Image Conversion:** Images are read and converted to NumPy arrays using OpenCV and resized to 256x256 pixels.
- **Normalization:** Pixel values are normalized by dividing by 255 to scale them between 0 and 1.
- **Data Splitting:** The dataset is split into training (80%), validation (20%), and testing (20%) subsets.

### 2. Model Architecture

A Convolutional Neural Network (CNN) is constructed with the following layers:

- **Conv2D Layer:** 32 filters with a kernel size of (3, 3), activation function ReLU.
- **MaxPooling2D Layer:** Pool size of (3, 3) for downsampling.
- **Conv2D Layer:** 16 filters with a kernel size of (3, 3), activation function ReLU.
- **MaxPooling2D Layer:** Pool size of (2, 2) for further downsampling.
- **Flatten Layer:** Flattens the 3D output to 1D.
- **Dense Layer:** Fully connected layer with 8 neurons and ReLU activation.
- **Dense Layer:** Output layer with 3 neurons (one for each class) and softmax activation.

### 3. Compilation and Training

- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam with a learning rate of 0.0001
- **Metrics:** Accuracy
- **Training:** The model is trained for 50 epochs with a batch size of 128. The training dataset is further split into training and validation sets.

### 4. Evaluation

- **Test Accuracy:** The model achieves a test accuracy of 97.22%.
- **Visualization:** Training and validation accuracy are plotted to visualize the training process.

## Results

The CNN model demonstrates high performance in classifying plant diseases with a test accuracy of 97.22%. This high accuracy indicates that the model effectively learned to differentiate between the different diseases based on the provided images.

Sample output from the model:

- **Original Image:** Potato-Early Blight
- **Predicted Label:** Potato-Early Blight

## Usage

To use this model for your own plant disease classification tasks, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/plant-disease-classification.git`
2. Install the required libraries: `pip install -r requirements.txt`
3. Place your dataset in the appropriate directory structure as described.
4. Obtain the model structure from the `leaf-disease-prediction-using-cnn.ipynb` file.

## Contributing

Feel free to contribute to this project by suggesting improvements, fixing bugs, or adding new features. Please follow the standard GitHub pull request workflow.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

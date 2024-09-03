## 3D Segmentation Model on CT Abdomen Organs

## 1. Overview

This project aims to implement and train a 3D segmentation model using the VNet architecture to segment organs from medical imaging data. 
The dataset used is the FLARE22 dataset, which consists of 3D MRI scans and corresponding ground truth segmentation masks. 
The goal is to train a model that can accurately segment multiple organs, including the liver, right kidney, left kidney, and spleen. 
The project includes data extraction, preprocessing, model building, training, evaluation, and visualization of results.

## 2. Setup Instructions
To set up the environment and run the code, follow these detailed instructions:

2.1. Prerequisites
Ensure you have the following software installed:

•	Python 3.7 or higher

•	TensorFlow 2.x

•	Nibabel

•	NumPy

•	Matplotlib

•	Scikit-learn

•	SciPy

You can install these packages using pip:
                         
                         !pip install tensorflow nibabel numpy matplotlib scikit-learn scipy


## 2.2. Downloading the Dataset
1. Download the FLARE22 dataset from the provided link.
   
2. Place the dataset zip file (e.g., FLARE22Train.zip) into your Google Drive or local directory.

## 2.3. Code Setup
1. Extract the Dataset:

The dataset is extracted if it has not been already. 
The code assumes that the dataset zip file is located at /content/drive/MyDrive/FLARE22Train.zip and will extract it to /content/FLARE22Train/FLARE22Train.

2. Update File Paths:

Ensure that the dataset_zip_path and extracted_path variables point to the correct locations of your dataset zip file and extraction directory.

3. Run the Code:

Execute the script in a Python environment. This script performs the following tasks:

• Extracts the dataset.

• Loads and preprocesses the data.

• Defines the VNet model architecture.

• Trains the model.

• Evaluates the model.

• Visualizes predictions.

## 3. Model Architecture

 3.1. VNet Architecture
 
VNet is a deep learning model designed for volumetric (3D) segmentation. 
It uses a similar approach to U-Net but is tailored for 3D image data. Below are key architectural details:

• Input Layer:

       • Takes a 3D input image with shape (128, 128, 64, 1).

• Encoder Path:

       • Convolutional Layers: Several 3D convolutional layers with ReLU activation functions to extract features.
       • MaxPooling Layers: Reduce the spatial dimensions of the feature maps.
• Bottleneck:

      • The deepest part of the network with several convolutional layers without downsampling.
 
• Decoder Path:

      • Transpose Convolutional Layers: Upsample the feature maps to the original spatial dimensions.
      • Concatenation: Concatenate features from the encoder path to preserve spatial details.

• Output Layer:


A 3D convolutional layer with a softmax activation function to predict the probability of each class for every voxel.

The model is built using TensorFlow/Keras and consists of multiple convolutional and pooling layers arranged in an encoder-decoder fashion. 
The output of the model is a 4D tensor where each voxel in the input image is classified into one of the predefined organ classes.

3.2. Dice Coefficient
The Dice coefficient is a metric used to evaluate the performance of the segmentation model. It measures the overlap between the predicted segmentation and the ground truth. The formula is:

     Dice = 2 × Intersection
           ------------------
           Sum of the sizes of two sets

In this implementation, the Dice coefficient is used both as a metric during model training and for evaluating performance on the validation set.

## 4. Training Process

4.1. Data Preprocessing

The preprocessing steps include:

1. Loading Data:

              The data is loaded using the nibabel library, which reads .nii.gz files.
2. Resizing:

   • Images and labels are resized to a fixed target dimension (128, 128, 64) using scipy.ndimage.zoom. This step ensures that all input data is of the same size.

3. Filtering Labels:
   
   • Labels are filtered to include only the target organ classes (liver, right kidney, left kidney, spleen). Non-target classes are excluded or set to zero.

4. Normalization:

• Images are normalized to the range [0, 1].

5. One-Hot Encoding:

• Labels are converted to one-hot encoded format using tf.keras.utils.to_categorical.

4.2. Training Procedure

1. Data Splitting:

• The dataset is split into training and validation sets using train_test_split from "sklearn."

2. Data Generators:

• A custom DataGenerator class is used to load and preprocess data on-the-fly during training. This class handles batching, shuffling, and data augmentation.

3. Model Compilation:

The model is compiled with the Adam optimizer, categorical crossentropy loss, and accuracy and Dice coefficient as metrics.

4. Model Training:

• The model is trained for 10 epochs. During training, the loss and accuracy are monitored for both training and validation data.

## 5. Validation and Inference

5.1. Validation Process

1. Validation Data:

• The validation set is used to evaluate the model's performance after training. Predictions are generated for the validation set.

2. Dice Coefficient Calculation:

• The Dice coefficient is calculated for each organ class separately. This metric evaluates the model's accuracy in segmenting each organ.

3. Visualization:

• The model's predictions are compared with the ground truth labels to visualize the segmentation results. Example slices from the validation set are plotted to illustrate the model's performance.

5.2. Performance Metrics

• Dice Coefficient: Measures the overlap between predicted and true segmentations. Higher values indicate better performance.

## 6. 3D Visualization Video 



https://github.com/user-attachments/assets/abf98dd0-8a7a-46ff-8cb7-ea3b7f71880a


# X-Ray Classification Project using VGG16 Transfer Learning Model

## Introduction

This project aims to classify X-ray images into two categories: pneumonia/COVID-19, and normal. The classification is performed using a VGG16 transfer learning model, 
which has been pre-trained on a large dataset and fine-tuned for this specific task. Additionally, the project involves analyzing COVID-19/pnuemonia spots within X-ray images 
using GRADCAM visualizations.

![image](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-022-27266-9/MediaObjects/41598_2022_27266_Fig1_HTML.png)

## Project Structure

The project is structured as follows:

1. **Data Collection and Preprocessing:**
   - Collect X-ray images of pneumonia, COVID-19, and normal cases.
   - Grouping covid-19/pnuemonia images together
   - Preprocess the images by resizing, normalizing, and augmenting the dataset.

2. **Model Building:**
   - Utilize the VGG16 architecture as a base model.
   - Fine-tune the model's top layers for the classification task.
   - Train the model using the preprocessed dataset.

3. **Model Evaluation:**
   - Evaluate the model's performance on a test set.
   - Calculate accuracy, precision, recall, and F1-score.

4. **GRADCAM Visualization:**
   - Implement GRADCAM technique to visualize important regions in COVID-19 positive X-ray images.
   - Generate heatmaps highlighting areas indicative of COVID-19 presence.

5. **Results and Analysis:**
   - Present the classification results including accuracy achieved on the test set (87%).
   - Display GRADCAM visualizations for a subset of COVID-19 positive images.

6. **Conclusion:**
   - Summarize the project's outcomes, insights, and limitations.
   - Discuss potential improvements and future work.

## Technologies Used

- Python
- TensorFlow (or other deep learning framework)
- ImageData Generator (for image preprocessing)
- GRADCAM visualization techniques

## Implementation Steps

1. Data Collection and Preprocessing:
   - Download datasets containing pneumonia, COVID-19, and normal X-ray images.
   - Resize images to a consistent size (e.g., 224x224) for compatibility with VGG16.
   - Normalize pixel values to the range [0, 1].
   - Augment the dataset by applying transformations like rotation and horizontal flip.

2. Model Building:
   - Load the pre-trained VGG16 model without its top layers.
   - Add custom fully connected layers for classification on top of the VGG16 base.
   - Compile the model with appropriate loss function and optimizer.

3. Model Training:
   - Split the dataset into training, validation, and test sets.
   - Train the model using the training set while validating on the validation set.
   - Apply early stopping to prevent overfitting.

4. Model Evaluation:
   - Evaluate the trained model on the test set.
     
5. GRADCAM Visualization:
   - Implement the GRADCAM algorithm to generate heatmaps.
   - Apply the algorithm to selected COVID-19-positive X-ray images.
   - Overlay the generated heatmaps onto the original images.


## Conclusion

This project demonstrates the successful utilization of a VGG16 transfer learning model for X-ray image classification, 
achieving an accuracy of 87% on the test set. Additionally, the application of GRADCAM visualization provides valuable insights into the areas 
indicative of COVID-19 presence within X-ray images. The project contributes to the field of medical imaging analysis and could potentially aid healthcare professionals
in accurate and efficient disease diagnosis.

![Banner](https://drive.google.com/uc?export=view&id=16KgGYux4NvdY28Mz0uHFbVJ14H1-pepV)


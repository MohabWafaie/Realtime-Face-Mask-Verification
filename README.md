# Realtime Face Mask Verification
Realtime Face Mask verification using Deep Convolutional Neural Networks and Haar Cascade
## About The Dataset
![dataset-cover](https://user-images.githubusercontent.com/39447236/218705098-1b8e247b-dfde-48cb-b04c-5a75f65f8010.png)


### Link
https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

### Context
This dataset is used for Face Mask Detection Classification with images. The dataset consists of almost 12K images which are almost 328.92MB in size  
10k images for Training  
800 images for Validation  
1000 images for Testing

### Acknowledgments
All the images with the face mask (~6K) are scrapped from google search and all the images without the face mask are preprocessed from the CelebFace dataset created by Jessica Li (https://www.kaggle.com/jessicali9530). Thank you so much Jessica for providing a wonderful dataset to the community

### Inspiration
The inspiration behind creating this dataset is to create an algorithm that can directly detect is a person is wearing a face mask or not. So I've scrapped the images from google as well as from the CelebFace dataset created by Jessica Li (https://www.kaggle.com/jessicali9530) to make this happen

## Image Preprocessing
1- Apply image augmentation to change the brightness range and orientation of images.  
2- Normalize the pixel values of images to between 0 and 1  
3- Reshape all images to be of size (256, 256, 3)  
All of this was done using Keras ImageDataGenerator class

## CNN Model
1- The model contains 4 CNN layers with number of filter ranging from 16 in the first layer to 64 in the last layer all with kernel size (3, 3) and Relu activation function  
2- Each layer is followed by a (2, 2) Max Pooling layer and a 20% Dropout layer   
3- Then followed by a Flattening layer  
4- Then a Dense layer of 128 neurons and Relu activation function  
5- A 20% Dropout layer  
6- An Output layer with Sigmoid activation function (since binary classification)  
  
![Screenshot 2023-02-14 121527](https://user-images.githubusercontent.com/39447236/218708040-eb54b410-15f6-450a-9c2f-c578d5183bee.png)   

## Model Compilation and Training
1- The model was compiled with Adam optimizer and Binary Crossentropy loss function  
2- The model was fit to the training data and achieved 99% accuracy on Validation data  
3- The model was evaluated on Testing data and achieved 98.3%    
![Screenshot 2023-02-14 123702](https://user-images.githubusercontent.com/39447236/218711311-1e6fafce-d9c5-48b3-8a28-043b256eb0a4.png)

## Realtime Implementation
1- Read frames from camera using OpenCV  
2- Extract faces using Haar Cascade  
3- Crop the faces from the frames  
4- Preprocess the faces to be suitable to be fed into the model  
5- Feed the faces to the model to predict whether it's with mask or without  
6- Show the output of predictions on the frames  
NOTE : These steps was done on 1 frame every 5 frames for faster processing

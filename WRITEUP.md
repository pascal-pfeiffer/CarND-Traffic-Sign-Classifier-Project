# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/train.png "Sample of train set"
[image2]: ./examples/valid.png "Sample of validation set"
[image3]: ./examples/test.png "Sample of test set"
[image4]: ./examples/class_counts.png "Class distribution"
[image5]: ./examples/augmented.png "Augmented images"
[image6]: ./examples/additional_test.png "Additional test images"
[image7]: ./examples/class_accuracy.png "Class accuracy"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/pascal-pfeiffer/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3) (width, height, channels)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First, I visualized samples of the train, validation and test portion with their corresponding labels:  

![Sample of train set][image1]
![Sample of validation set][image2]
![Sample of test set][image3]

The images are very different in terms of lighting and visibility, but there seems to be no significant difference between train, validation and test.  

Next, I plotted the class distribution of the training set in a horizontal bar chart.

![Class distribution][image4]

It is clearly visible, that we have a highly unbalanced training set which may cause problems in the training procedure. Some signs only appear about 200 times in the train set, whereas other appear almost 2000 times. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To prevent overfitting and to takle the rather small dataset, I decided to generate additional data by using augmentation techniques. I used moderate augmentation including rotation, shear, zoom and shifts in x and y direction.

Here are some examples of augmented images:

![Augmented images][image5]

Finally, I decided to normalize the images on a sample and channel (RGB) basis to uniform variance and zero mean.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consists of the following layers:

| Layer					|     Description								| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, output 32x32x64 	|
| ReLU					|												|
| Max pooling			| 2x2 stride, output 16x16x64 					|
| Convolution 3x3		| 1x1 stride, same padding, output 16x16x64		|
| ReLU					|												|
| Max pooling			| 2x2 stride, output 8x8x64 					|
| Flatten				| 2x2 stride, output 8x8x64 					|
| Dropout				| 20 % change for dropout 						|
| Fully connected		| output 120 									|
| ReLU					|												|
| Dropout				| 20 % change for dropout 						|
| Fully connected		| output 84 									|
| ReLU					|												|
| Dropout				| 20 % change for dropout 						|
| Fully connected		| output 43      								|
| Softmax				| 												|

Trainable params: 544,179

The models is an adapted version of LeNet with additional dropout layers and 43 outputs for each of the classes in the set. 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a standard learning rate of 0.001. Batchsize was chosen to 64, and the training was done for 10 epochs. To enhance accuracy an ensemble of 5 single models was trained and the softmax predictions are then averaged across all models. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The LeNet architecture was chosen at it proved to be able to solve tasks such as image classification. Convolutions help the Neural Network to lern features in the 2D image. The single models each have an accuracy of ~94-95 % on the validation and test set. Due to the data augmentation and dropout, the training accuracy is not much higher than validation accuracy. That means, overfitting was prevented sucessfully. 

| Model				| Training Accuracy		| Validation Accuracy	| Test Accuracy			|
|:-----------------:|:---------------------:|:---------------------:|:---------------------:|
| 1					| 0.9607				| 0.9540				| 0.9421				|
| 2					| 0.9673				| 0.9497	 	 	 	| 0.9488				|
| 3					| 0.9638				| 0.9694				| 0.9597				|
| 4					| 0.9671				| 0.9603				| 0.9536				|
| 5					| 0.9644				| 0.9635				| 0.9503				|
| ensemble			| 						| 						| **0.9731**			|

The accuracy of the ensemble of those 5 models reaches more than 97 % on the test set. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Additional test images][image6]

The first three images (bumpy road, yield, stop) should be rather easy to classify as they show destinct markers and no occlusion. For the forth sign (no entry), i've chosen an image with multiple signs. The model may be tricked by this. The last sign (wild animals crossing) is heavily affected by the downscaling to 32x32 pixels. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction				| 
|:---------------------:|:-----------------------------:| 
| Bumpy road      		| Bumpy road   					| 
| Yield     			| Yield 						|
| Stop					| Stop							|
| No entry	      		| No entry					 	|
| Wild animals crossing	| Wild animals crossing      	|


The model was able to correctly guess all of the traffic signs, which gives an accuracy of 100 %. This compares favorably to the accuracy on the test set of which reached more than 97 %. 

On the provided test data I checked the accuracy of each class. 

![Class accuracy][image7]

For most classes, the accuracy reaches a perfect 100 %. Only "Double curve" and "Pedestrians" shows an accuracy below 80 %. Both, signs with a rather low count.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

By looking at the softmax output of the model we can get an estimate of how sure the model is in its predictions. For the first four images, the model is relatively certain (probability above 99.999 %), and correctly classifies the images. 

For the last image, which was a "Wild animals crossing", the certainty drops to 82.86 %. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8.28649640e-01  	 	| Wild animals crossing   						| 
| 1.71267539e-01  		| Slippery road 								|
| 5.19492605e-05		| Dangerous curve to the left					|
| 1.26056793e-05		| Double curve					 				|
| 1.01309715e-05		| No passing for vehicles over 3.5 metric tons	|
 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



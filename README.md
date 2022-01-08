# Estimation-of-Food-Calories

**Introduction:**

* Due to the improvement in people’s standards of living, obesity rates are increasing at an alarming speed. People need to control their daily calorie intake by eating healthier foods. However, although food packaging comes with nutrition (and calorie) labels, due to the difficulties and complexity to record and track all the ingredients of a meal,it’s still not very convenient for people to reference. Thus, scientists started to use machine learning algorithms in computer vision to help people determine the caloric value in the food they eat. 

* In the industry, the ordinary process to do calories detection contains 2 major steps. Food-item identification and food calories estimation.

  - *Food-item Identification:* To identify what’s on the plate, we need to instance-segment the given food image into the possible food categories. Mask R-CNN would be a matching solution to instance segmentation. Mask R-CNN takes an image and spits out three outputs, masks of the identified items, bounding boxes and classes for each mask detected.

  - *Food Calorie Estimation:* As the same food can be taken at different depths to generate different picture sizes. we need a method to calculate calorie or estimate the size of the food in a real-world scenario. After we get the desired food items detected along with their masks, we need the real object sizes. So, we take a referencing approach that references the food-objects to the size of the pre-known object to extract the actual size of the food contained in that specific image.

**Related Work:**
* In the original paper that introduced the dataset used in this project, Bossard  employed a weakly supervised mining method that relied on Random Forests (RFs) to mine discriminative regions in images(accuracy of 50.76%)
* A subsequent study on food image classification focused solely on the use of CNNs constructed a five-layer CNN to recognize a subset of ImageNet data, which consisted of ten food classes(accuracy of 74%)
* More recently, Liu implemented DeepFood, a CNN-based approach inspired by LeNet-5, AlexNet, and GoogleNet, employing Inception modules to increase the overall depth of the network structure. (accuracy of 77.4%)

**Project Goal:**
* The goal of the project is to, given an image of a dish as the input to the model, output the correct label categorization of the food image.
* To simplify industry model, without instance segmentation in a meal, we used single food category in a image instead of mixed food categories. Besides, not taking reference approach to extract actual size of the food, we utilized the unit calories. Therefore our method is simply food classification with unit label attached.

**Dataset:** 
* There are labeled food images in 101 categories from apple pies to waffles, with 1000 images, that have a resolution of 384x384x3 (RGB, uint8). Of the 1000 images for each class, 250 were manually reviewed test images, and 750 were intentionally
noisy training images,
* Take 10 categories :cheese cake/ chicken wings/ donuts/ fried rice/ french fries/ gyoza/ ice-cream/ oyster/ scallops/ waffle
* Those 10 categories vary from the size and the color and even the food category. We use them as a perfect sample of food dataset.

**Models:** 

(1)	CNN 2D:
* Convolutional Neural Network is very good at image classification for mainly two reasons:
  1.	No need for manual feature extraction: A CNN convolves learned features with input data, and uses 2D convolutional layers. How CNN work is by extracting features directly from images and the key features are not pretrained; they are learned while the network trains on a collection of images. It is the automated feature extraction that makes CNNs highly suited for and accurate for computer vision tasks such as image classification. 
  2.	Dimensionality reduction: CNNs are very effective in reducing the number of parameters without losing on the quality of models. Images have high dimensionality (as each pixel is considered as a feature) which suits the above described abilities of CNNs. 
* test_acc: 0.0833333358168602

(2)	Add dropout layer:
* In case of overfitting, dropout is placed on the fully connected layers because they are the one with the greater number of parameters and thus they're likely to excessively co-adapting themselves causing overfitting.
* By adding dropout layer, we not only highly reduce any possible overfitting but also increased the performance of our CNN a little bit.
* test_acc with Dropout layer: 0.15000000596046448

(3) Hyperparameters:
* Activation function: ReLU
  - In order to use stochastic gradient descent with backpropagation of errors to train deep neural networks, an activation function is needed that looks and acts like a linear function, but is, in fact, a nonlinear function allowing complex relationships in the data to be learned.
  - The function must also provide more sensitivity to the activation sum input and avoid easy saturation. The solution is to use ReLU.

* For the output layer, we chose ‘softmax’ as the activation function. It can predict a multinomial probability distribution so it’s usually used in multi-class classification problems.

* Loss function: categorical_crossentropy
  - It can compute the crossentropy loss between the labels and predictions.
  - It is usually used when there are two or more label classes so we use it here. 

(3)	Data augmentation
* The benefits of data augmentation are two: The first is the ability to generate ‘more data’ from limited data. The second one is to avoid overfitting: This occurs because the model memorizes the full dataset instead of only learning the main concepts underlying the problem.

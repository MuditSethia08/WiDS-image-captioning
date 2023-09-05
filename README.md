# WiDS-Image-Caption-Generation

The aim of this project is to generate captions for images, which describe the image,  using Deep Learning. Final model used VGG19 architecture combined with LSTM and NLP for more better results.

## Week 1
- Started with learning basic python libraries such as Pandas & numpy for data analysis
- Studied Exploratory Data Analysis by using various types of Graphs and Figures to depict data
- Learnt basic Machine Learning prediction techniques such as Decision Tree, Random Forest etc
- Important data pre-processing to deal with Missing values (Imputataion) or Categorical variables (Ordinal / One Hot Encoding) or using Pipelines
- Assignment task was to predict individual product failures of new codes using various ML algorithms with evaluation based on AUC under ROC curve obtained
- Implemented algorithms like XGBoost, KNN Classifier, Decicion Tree, Logistics Regression and Random Forest for predictions along with some data-preprocessing techniques

## Week 2
- Started learning Neural Networks and concepts of back-tracking, activation functions, and hyperparameter tuning
- Assignment was on building a classifier for MNIST dataset using PyTorch 
- Trained a Neural Network for recognizing hand-written digits from MNIST
- Used Linear layers, Batch Normalization and Rectified Linear Unit in the Network
- Final Accuracy achieved : $96.85$ % 

## Week 3
- Started learning theory of Convolutional Neural Networks and their architectures
- Used CNNs in PyTorch for classifying images from **CIFAR-10** dataset
- Used several layers such as BatchNorm2d, Fully Connected, ReLU, Pooling, Linear and Conv2d in the architecture of ConvNet
- Final Accuracy achieved : $57.91$ % 

## Week 4
- Learnt Recurrent Neural Networks (RNN), Long Short Term Memory (LSTM) and Natural Language Processing in this week
- Worked on 2 Assignments : 
  1. Predicting future Oil prices for next 30 days
  2. Sentimental Analysis on Stock Market statements
- Trained LSTM model for predicting future oil prices based on current historical data
- Implemented NLP techniques of stopword removal, Tokenization and Lemmaization in sentiment prediction of sentences

## Week 5
- Final week consisted of using all the concepts leant to build a model that generates captions for images
- Used *Flickr8k* dataset for training and testing the model, which consists of 8000 images with 5 captions for each image
- Utilized pre-trained **VGG-19** neural network for extracting features from each image 
- Passed these features into the **LSTM** network for generating captions
- Evaluated the model using **BLEU** by calculating BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
# CreditCard Fraud Detection System using ML
 
Using different Machine Learning techniques, trying to detect Fraudulent Credit Card transactions. Comparing the performances of all the models used.

In this notebook, I explore various Machine Learning models to detect fraudulent use of Credit cards. I compare each model performance and results. The best performance is achieved using SMOTE technique.


# Problem Statement

In this project we want to identify fraudulent transactions with Credit Cards.
Our objective is to build a Fraud detection system using Machine learning techniques.

The project uses a dataset of 300,000 fully anonymized transactions. Each transation is labelled either fraudulent or not fraudulent.
Note that prevalence of fraudulent transactions is very low in the dataset. Less than 0.1% of the card transactions are fraudulent. This means that a system predicting each transaction to be normal can reach an accuracy of over 99.9% despite not detecting any fraudulent transaction. This will necessitate adjustment techniques.

# Techniques used in the project
The project compares the results of different techniques :
- Machine learning techniques:
  - Random Forest
  - Decision Trees
  - Gradient Boosting Classifier
- Deep Learning techniques:
  - Neural network using fully connected layers.
  - Convolution Neural Network (conv1D)

Performance of the neural network is compared for different optimization approaches:
- plain binary cross-entropy loss minimization
- minimization using weights to compensate for the class imbalance
- Under-sampling of the non-fraudulent class to match the fraudulent class
- Over-sampling of the fraudulent class to match the non-fraudulent one by implementing SMOTE technique. The SMOTE method allows to generate a new vector using 2 existing datapoints. For additional details on this approach, you can read this detailed post [SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)

CNN model performs better than all the other models. The training in this model is faster and the model metrics achieved are the best. Used a basic CNN and trained for just 5 epochs. This can be scaled to any number depending on the size of the data.

- **Normalizing the data after splitting the dataset to avoid unneccesary leakages of information from test set into the training process.** 

# Results

The best results are achieved by over-sampling the under-represented class using SMOTE (synthetic minority oversampling technique).
With this approach, the model is able to detect 100% of all fraudulent transactions in the unseen test set. This fully satisfies the primary objective to detect the vast majority of abnormal transactions. Please note that the technique and model used are simple to implement simple, easy to use and can be updated in real-time.

In addition, the number of false positive remains acceptable. This means a lot less verification work (on legitimate transactions) for the fraud departement compare dto some other approaches which failed on this aspect. Key results are shown below:

The CNN model is evaluated using the Keras classifier found in the keras wrappers for scikit_learn. Using the sklearn metrics (KFold cross validation) the model performance is calculated by extracting the cross validation score for each fold. A pipeline was build to process the whole data in each step.




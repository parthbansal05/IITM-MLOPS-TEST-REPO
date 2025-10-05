import pandas as pd 
import joblib


iris = pd.read_csv("data.csv")

print(iris.head())


# importing alll the necessary packages to use the various classification algorithms
from sklearn.model_selection  import train_test_split # to split the dataset for training and testing 
from sklearn import svm # for suport vector machine algorithm
from sklearn import metrics # for checking the model accuracy

train, test = train_test_split(iris, test_size=0.3) # our main data split into train and test


train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking the training data features
train_y = train.Species # output of the training data

test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data feature
test_y = test.Species # output value of the test data


model = svm.SVC() # select the svm algorithm

# we train the algorithm with training data and training output
model.fit(train_X, train_y)

# we pass the testing data to the stored algorithm to predict the outcome
prediction = model.predict(test_X)
print('The accuracy of the SVM is: ', metrics.accuracy_score(prediction, test_y)) # we check the accuracy of the algorithm
#we pass the predicted output by the model and the actual output


filename = 'svm_iris_model.joblib'

# 2. Use joblib.dump() to save the trained model object to a file
joblib.dump(model, filename)

print(f"\nModel successfully saved as: {filename}")
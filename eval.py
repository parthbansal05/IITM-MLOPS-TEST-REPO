import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 

model = joblib.load('svm_iris_model.joblib')

csv_file_path = 'data.csv'
data = pd.read_csv(csv_file_path)
print(data.head())
_, test_data = train_test_split(data, test_size=0.3, random_state=42)

test_X = test_data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_y = test_data.Species

prediction = model.predict(test_X)

accuracy = metrics.accuracy_score(prediction, test_y)
print('Model loaded and evaluated.')
print('The accuracy on the test set is:', accuracy)
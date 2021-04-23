# import comet_ml at the top of your file
from comet_ml import Experiment

#Create an experiment with your api key
experiment = Experiment(
    api_key="5ygTg5xsKwnceT4oUw8huVacF",
    project_name="test1",
    workspace="ih127",
)

# Run your code and go to /
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf = MLPClassifier(random_state = 1, max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print(classification_report(y_test, preds))
acc = accuracy_score(y_test, preds)
experiment.log_metric("accuracy", acc)

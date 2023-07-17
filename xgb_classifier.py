import keras.utils as np_utils
import numpy as np
import sklearn.metrics as metric

from feature_extractor import FeatureExtractor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

feature_extractor = FeatureExtractor()

labels, features = feature_extractor.extract_features_from_csv()

# Creates a 2-dimensional array
features = np.array(features.tolist())
labels = np.array(labels.tolist())

# Splits given dataset into test and train groups
x_train, x_test, y_train, y_test = train_test_split(features, labels)

# Encodes labels to numbers. For example, "dog", "cat", "bear" would become 1,2,3
encoder = LabelEncoder()
y_train = np_utils.to_categorical(encoder.fit_transform(y_train))
y_test = np_utils.to_categorical(encoder.fit_transform(y_test))

# Here, we are using XGBClassifier as a Machine Learning model to fit the data
train_y = np.argmax(y_train, axis=1)
test_y = np.argmax(y_test, axis=1)
train_x = x_train
test_x = x_test

model = XGBClassifier(learning_rate=0.1,
                      n_estimators=280,
                      max_depth=5,
                      min_child_weight=1,
                      gamma=0,
                      subsample=0.8,
                      colsample_bytree=0.8,
                      objective='binary:logistic',
                      nthread=4,
                      seed=60)
print(model.fit(train_x, train_y))
predict = model.predict(test_x)

# Printed classification report and confusion matrix for the XGBclassifier
print(f'Model Score: {metric.accuracy_score(test_y, predict)}')
print(f'Confusion Matrix: \n{metric.confusion_matrix(test_y, predict)}')

import numpy

from feature_extractor import FeatureExtractor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

feature_extractor = FeatureExtractor()

labels, features = feature_extractor.extract_features_from_csv()

# Creates a 2-dimensional array
features = numpy.vstack(features)

# Encodes labels to numbers. For example, "apple", "brownie", "cheeseburger" would become 1,2,3
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Splits given dataset into test and train groups
x_train, x_test, y_train, y_test = train_test_split(features, encoded_labels)

# The 'n_neighbors' determines the number of neighbors considered by the K-nearest neighbors algorithm. Grid search
# systematically evaluates different hyperparameter values to find the combination that yields the best performance
# based on a chosen evaluation metric. 'weights' parameter specifies the weight function used in prediction.
# 'metric' parameter defines the distance metric used to calculate the distances between instances.
# It can take two possible values: 'euclidean' (which uses Euclidean distance) and 'manhattan'
# (which uses Manhattan distance). The grid search will try both of these distance metrics.
grid_params = {
    'n_neighbors': range(1, 20),
    'weights': ['uniform'],
    'metric': ['euclidean', 'manhattan']
}

# Preprocessing class used to standardize features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
# This step computes the mean and standard deviation of each feature in the training data.
scaler.fit(x_train)
# This step applies the computed mean and standard deviation to scale (standardize) the training data.
# It subtracts the mean from each feature and divides by the standard deviation.
# This ensures that the features have zero mean and unit variance.
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# An instance of the PCA class from sklearn.decomposition is created. PCA stands for Principal Component Analysis,
# a technique for dimensionality reduction.The fit() method of the PCA instance is called with
# x_train_scaled as the argument. This step computes the principal components from the standardized training data.
pca = PCA().fit(x_train_scaled)

# Grid search is used for hyperparameter tuning: finding the best hyperparameters for model training.
model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
model.fit(x_train_scaled, y_train)

print(f'Model Score: {model.score(x_test_scaled, y_test)}')

# This line generates predictions for the test data using the trained model.
y_predict = model.predict(x_test_scaled)
print(f'Confusion Matrix: \n{confusion_matrix(y_predict, y_test)}')

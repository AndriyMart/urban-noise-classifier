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

features = numpy.vstack(features)

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Since the data manufacturer doesn't provide the labels for the test audios,
# we will have do the split for the labeled data.

x_train, x_test, y_train, y_test = train_test_split(features, encoded_labels)

grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

pca = PCA().fit(x_train_scaled)

model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
model.fit(x_train_scaled, y_train)

print(f'Model Score: {model.score(x_test_scaled, y_test)}')

y_predict = model.predict(x_test_scaled)
print(f'Confusion Matrix: \n{confusion_matrix(y_predict, y_test)}')

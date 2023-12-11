from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from dataset_preparer import get_dataset

train_ds = get_dataset('train', in_memory=True)
validation_ds = get_dataset('validation', in_memory=True)
test_ds = get_dataset('test', in_memory=True)

train_images, train_labels = train_ds
validation_images, validation_labels = validation_ds
test_images, test_labels = test_ds

def evaluate_model_performance(groundtruth, prediction):
    accuracy = accuracy_score(groundtruth, prediction)
    print(f'Test Accuracy: {accuracy:.2%}')

    print("Classification Report:")
    print(classification_report(groundtruth, prediction))

# kNN
k_neighbors = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)
knn_classifier.fit(train_images, train_labels)

prediction = knn_classifier.predict(test_images)
evaluate_model_performance(test_labels, prediction)

# NaiveBayes
nb_classifier = GaussianNB()
nb_classifier.fit(train_images, train_labels)

prediction = nb_classifier.predict(test_images)
evaluate_model_performance(test_labels, prediction)

# RandomForest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(train_images, train_labels)

prediction = rf_classifier.predict(test_images)
evaluate_model_performance(test_labels, prediction)

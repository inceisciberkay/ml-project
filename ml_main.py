from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from dataset_preparer import get_dataset
from sklearn.model_selection import cross_val_score
import numpy as np
from evaluation_utils import plot_confusion_matrix, calculate_evaluation_metrics

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
    cm = confusion_matrix(groundtruth, prediction)
    TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    accuracy, precision, recall, f1_score, f2_score = calculate_evaluation_metrics(TP, FP, FN, TN)
    result = f"""\
TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1 Score: {f1_score}
F2 Score: {f2_score}
"""
    print("Evaluation Metrics:")
    print(result)
    plot_confusion_matrix(TP, FP, FN, TN)

# merge train and validation
# train_images = np.concatenate((train_images, validation_images), axis=0)
# train_labels = np.concatenate((train_labels, validation_labels), axis=0)

# NaiveBayes
print('\n=======Naive Bayes=======\n')
nb_classifier = GaussianNB()

nb_classifier.fit(train_images, train_labels)
predictions = nb_classifier.predict(test_images)
evaluate_model_performance(test_labels, predictions)

# kNN
# perform 4-fold cross validation to find the best k
k_neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
k_scores = []
for k in k_neighbors:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_classifier, train_images, train_labels, cv=4)
    k_scores.append(np.mean(scores))
print("4-fold cross validation scores: ", k_scores)
best_k = k_neighbors[np.argmax(k_scores)]
print("Best k: ", best_k)

# use the best k to improve the accuracy
knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(train_images, train_labels)
prediction = knn_classifier.predict(test_images)
evaluate_model_performance(test_labels, prediction)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(TP, FP, FN, TN):
    # Create the confusion matrix
    conf_matrix = np.array([[TN, FP],
                            [FN, TP]], dtype='float')

    conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
    class_names = ['Nonfractured', 'Fractured']

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format=".4f")
    plt.title('Confusion Matrix')
    plt.show()

def print_evaluations(TP, FP, FN, TN):
    # Accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN) * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Precision
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    print(f'Precision: {precision:.2f}')

    # Recall
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    print(f'Recall: {recall:.2f}')

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print(f'F1 Score: {f1_score:.2f}')

    # F2 Score
    beta = 2
    f2_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if ((beta**2 * precision) + recall) != 0 else 0
    print(f'F2 Score: {f2_score:.2f}')

# metrics = [
#     # (2608, 103, 154, 1221),
#     # (2594, 103, 168, 1221),
#     # (2554, 74, 208, 1250),
#     # (2488, 79, 274, 1245),
#     # (2391, 51, 371, 1273),
#     # (2625, 96, 137, 1228),
#     # (2431, 67, 331, 1257),
#     # (2450, 78, 312, 1246),
#     # (2481, 85, 281, 1239),
#     # (2488, 79, 274, 1245),
#     # (2518, 103, 244, 1221),
#     # (2544, 98, 218, 1226),
#     # (1989, 7, 773, 1317)
#     # (2572, 129, 190, 1195)
# ]

# for metric in metrics:
#     plot_confusion_matrix(*metric)

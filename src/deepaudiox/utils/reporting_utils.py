import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def get_classification_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_mapping: dict
):
    """Create a classification report based ot true and predicted labels.

    Args:
        y_true (np.ndarray): A NumPy array with the true labels.
        y_pred (np.ndarray): A NumPy array with the predicted labels.
        class_mapping (dict): A mapping between class names and IDs.

    Returns:
        dict: Classification report
    
    """

    target_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]

    return classification_report(
        y_true=y_true, 
        y_pred=y_pred, 
        target_names=target_names,
        zero_division=0
    )

def get_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_mapping: dict
):
    """Create a confusion matrix based ot true and predicted labels.

    Args:
        y_true (np.ndarray): A NumPy array with the true labels.
        y_pred (np.ndarray): A NumPy array with the predicted labels.
        class_mapping (dict): A mapping between class names and IDs.

    Returns:
        np.ndarray: Confusion matrix
    
    """
    target_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]

    return confusion_matrix(
        y_true=y_true, 
        y_pred=y_pred, 
        labels=list(range(len(target_names)))
    )

def get_avg_posteriors(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    posteriors: np.ndarray,
    class_mapping: dict
):
    """Calculate the average posterior per class.

        Args:
            y_true (np.ndarray): A NumPy array with the true labels.
            y_pred (np.ndarray): A NumPy array with the predicted labels.
            posteriors (np.ndarray): A NumPy array with the posterior probabilities.
            class_mapping (dict): A mapping between class names and IDs.

        Returns:
            dict: Average posterior probability per class
    
    """
    mean_posteriors = {}
    target_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]
    for label in target_names:
        mask = (y_true == y_pred) & (y_pred == class_mapping[label])
        posteriors_of_correctly_classified = posteriors[mask]
        if len(posteriors_of_correctly_classified) > 0:
            mean_posteriors[label] = float(np.mean(posteriors_of_correctly_classified))
        else:
            mean_posteriors[label] = np.nan

    return mean_posteriors
import numpy as np
from BayesMethod import learn_cpts, predict_label, Bayesian_probabilities, combine_probabilities
from sklearn.metrics import accuracy_score


def tune_lambda_value(x_valid, y_valid, cpts, dl_probs_valid, lambda_values):
    """
    Tune the 'lambda_value' hyperparameter on the validation set.
    
    Parameters:
    - x_valid (numpy.ndarray): The input features for the validation set.
    - y_valid (numpy.ndarray): The true labels for the validation set.
    - cpts (dict): The learned Conditional Probability Tables.
    - dl_probs_valid (numpy.ndarray): The deep learning probabilities for the validation set.
    - lambda_values (list of float): A list of 'lambda_value' to consider.
    
    Returns:
    - float: The optimal 'lambda_value'.
    """
    best_lambda = 0
    best_accuracy = 0
    
    y_valid_seq = np.argmax(y_valid, axis=-1)
    bayesian_probs_valid = Bayesian_probabilities(cpts, y_valid_seq, y_valid.shape[1])
    
    for lambda_value in lambda_values:
        combined_probs_valid = combine_probabilities(dl_probs_valid, bayesian_probs_valid, lambda_value)
        predictions = np.argmax(combined_probs_valid, axis=1)
        accuracy = accuracy_score(y_valid_seq, predictions)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda = lambda_value
    
    return best_lambda



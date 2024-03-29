# -*- coding: utf-8 -*-
"""Bayesian-Method.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-uLeKFiX35MgycUS1JVLU7lgkVQTlYIx
"""


import numpy as np
import random
from collections import defaultdict, Counter



def learn_cpts(y, k):
    """
    Learn and return the Conditional Probability Tables (CPTs) for a given sequence of class labels.

    Args:
    y (list): An array of consecutive class labels L_1, L_2, ..., L_n representing the labels of consecutive windows.
    k (int): The maximum number of previous windows to consider.

    Returns:
    dict: A dictionary where keys are tuples representing conditions (previous labels) and values are Counter objects
    representing the frequency of current labels given the conditions.
    """

    # Initialize the dictionary to hold the CPTs
    cpts = defaultdict(Counter)

    # Iterate over the sequence of labels
    for i in range(len(y)):
        # Randomly choose the number of previous windows to consider, between 1 and k, ensuring it does not exceed the current index
        m = random.randint(1, min(k, i) if i > 0 else 1)

        # Extract the current label and the previous m labels
        current_label = y[i]
        previous_labels = tuple(y[max(0, i - m) : i])

        # Update the CPTs
        cpts[previous_labels][current_label] += 1

    # Convert counts to probabilities
    for previous_labels, counter in cpts.items():
        total = sum(counter.values())
        for label in counter:
            cpts[previous_labels][label] /= total

    return cpts

def predict_label(cpts, previous_labels, num_classes):
    """
    Predict the probability vector using a weighted average based on the length of each subsequence.

    Args:
    cpts (dict): A dictionary where keys are tuples representing conditions (previous labels) and values are Counter objects
    representing the frequency of current labels given the conditions.
    previous_labels (tuple): A tuple of previous labels.
    num_classes (int): The number of distinct classes.

    Returns:
    list: A probability vector of length num_classes.
    """
    # Initialize variables to hold the sum of weighted probabilities and the sum of weights
    weighted_prob_sum = [0] * num_classes
    weight_sum = 0

    for i in range(1, len(previous_labels) + 1):
        sub_sequence = tuple(previous_labels[-i:])
        if sub_sequence in cpts:
            sub_sequence_probs = cpts[sub_sequence]

            # Calculate the weight for this subsequence (e.g., proportional to its length)
            weight = len(sub_sequence)

            for label, prob in sub_sequence_probs.items():
                weighted_prob_sum[label - 1] += prob * weight

            weight_sum += weight

    # Avoid division by zero if no subsequences were found
    if weight_sum == 0:
        return [1 / num_classes] * num_classes

    # Calculate the weighted average probability vector
    weighted_average_prob_vector = [prob_sum / weight_sum for prob_sum in weighted_prob_sum]

    return weighted_average_prob_vector

def Bayesian_probabilities(cpts, sequence, num_classes):

    bayesian_probabilities = []

    # Iterate through each label in the sequence
    for i, label in enumerate(sequence):
        # Extract the previous labels for each instance
        previous_labels = sequence[:i]

        # Predict the probability vector using the CPTs
        probability_vector = predict_label(cpts, previous_labels, num_classes)

        # Append the predicted probabilities to the final list
        bayesian_probabilities.append(probability_vector)

    # Convert the list of predicted probabilities to a NumPy array for easy manipulation
    bayesian_probabilities = np.array(bayesian_probabilities)

    return bayesian_probabilities

def combine_probabilities(dl_probs, bayesian_probs, lambda_value):
    """
    Combine the probabilities from the deep learning model and Bayesian model using a weighted average and normalize them.

    Args:
    dl_probs (list): A list of probabilities from the deep learning model.
    bayesian_probs (list): A list of probabilities from the Bayesian model.
    lambda_value (float): A hyperparameter controlling the weight given to each model's probabilities.

    Returns:
    list: A normalized combined probability vector.
    """

    if not (0 <= lambda_value <= 1):
        raise ValueError("Lambda value should be between 0 and 1.")

    # Calculate the weighted probabilities
    combined_probs = [(lambda_value * dl_prob) + ((1 - lambda_value) * bayesian_prob)
                      for dl_prob, bayesian_prob in zip(dl_probs, bayesian_probs)]

    # Normalize the combined probabilities to sum up to 1
    total_prob = sum(combined_probs)
    normalized_probs = [prob / total_prob for prob in combined_probs] if (total_prob > 0).any() else combined_probs

    return normalized_probs
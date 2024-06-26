# -*- coding: utf-8 -*-
"""performance_comparison.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xpidRAVB7D_wdQ9fWcW9X5aYJ-3v3g4a
"""

import matplotlib.pyplot as plt
import numpy as np

datasets = ["UniMiB SHAR", "UCI HAR", "Leotta", "TWristAR"]
# Hypothetical data for box-and-whisker plot
# For each method, we'll assume 10 runs on each dataset, represented by the list of accuracies
# These are hypothetical numbers for demonstration purposes

our_method_data = [
    [90, 91.1, 91.5, 92.2, 92.1, 92.3, 92.5, 92.8, 92.9, 93.1],  # UniMiB SHAR
    [81.1, 81.2, 82.4, 82.8, 82.8, 82.7],  # UCI HAR
    [42.3, 43.4, 44, 45.2, 45.2, 43.4, 46, 46.5, 46.8, 47.1],  # Leotta
    [80.7, 78.4, 77.9, 77.5, 77.9, 78.5, 78],  # TWristAR


]

baseline_cnn_data = [
    [84.3, 84.6, 85.0, 85.5, 86.2, 87.0, 87.5, 87.8, 88.0, 88.2],
    [78.2, 79.0, 79.6, 79.8, 80.3, 81.1],
    [31.1, 33.5, 33.1, 34, 35, 36, 36.5, 36.7, 36.7, 36.7],
    [68.4, 69.4, 71.0, 71.8, 72.0, 73.4, 74.2],


]

InceptionTimePlus_data = [
    [84.7, 83.8, 84.3, 84.4, 83.2, 86.4, 84.9, 84.9, 87.6],
    [79.5, 79.6, 79.4, 78.8, 79.4, 78.6, 78.6, 78.1, 78],
    [47.1, 45.5, 45.4, 48.0, 47.5, 46.9, 45.2, 48.0, 46.7, 47.8],
    [76.0, 74, 74, 75, 72.1, 77.4, 76, 77.9, 75],


]

lstnet_data = [
    [88.3, 88.6, 89.0, 89.5, 90.2, 91.0, 91.5, 91.8, 91.9, 92.0],
    [79, 81.3, 80.6, 80.8, 80.9, 81,],
    [42.2, 43.3, 42.8, 42.9, 40.7, 40.9, 42.5, 41.3, 40.6],
    [76, 76.9, 73.1, 75],


]

TSTPlus_data = [
    [85.3, 85.8, 86.3, 88.2, 85.4, 86.3, 85.6 ],
    [81.3, 80.4, 80.0, 79.7, 79.4, 80.4, 80.3 ],
    [39.3, 37.5, 38.4, 38.0, 38.7, 38.8],
    [71.2, 74.0, 74.5, 73.6, 71.6, 72.2 ],


]


# Creating the box-and-whisker plot
plt.figure(figsize=(12, 6))

# Function to prepare data for side-by-side plotting
def prepare_data_for_side_by_side_plotting(*args):
    # This function will arrange data for side-by-side plotting
    grouped_data = []
    for i in range(len(args[0])):  # Iterate over each dataset
        for data in args:  # Iterate over each method
            grouped_data.append(data[i])
    return grouped_data

# Prepare data for side-by-side plotting
plot_data = prepare_data_for_side_by_side_plotting(our_method_data, baseline_cnn_data, InceptionTimePlus_data, lstnet_data, TSTPlus_data)

# Number of datasets and methods
n_datasets = len(datasets)
n_methods = 5  # Our Method, Baseline CNN, Baseline LSTM, LSTNet, TSTPlus

# Creating the box-and-whisker plot for side-by-side comparison
plt.figure(figsize=(14, 6))

# Plotting
base_positions = np.arange(1, n_datasets + 1)  # Base positions for each dataset
offset = np.linspace(-0.4, 0.4, n_methods)     # Fixed offsets for each method
colors = ['lightblue', 'lightgreen', 'lightcoral', 'wheat', '#9467bd']  # Colors for each method

# Plot each method's data and create custom legend patches
legend_patches = []
for i, method_data in enumerate([our_method_data, baseline_cnn_data, InceptionTimePlus_data, lstnet_data, TSTPlus_data]):
    pos = base_positions + offset[i]
    boxplot = plt.boxplot(method_data, positions=pos, widths=0.15, patch_artist=True, boxprops=dict(facecolor=colors[i]))
    legend_patches.append(plt.Line2D([0], [0], color=colors[i], linewidth=10, label=['Our Method', 'Baseline CNN', 'InceptionTime', 'LSTNet', 'TST'][i]))

# Adding vertical lines to separate datasets
for x in np.arange(1.5, n_datasets + 0.5, 1):
    plt.axvline(x=x, color='grey', linestyle='--', linewidth=0.5)

# Adding labels and title
plt.xlabel('Datasets')
plt.ylabel('Accuracy (%)')
plt.title('Box-and-Whisker Plot of Time-Series Prediction Accuracy')
plt.xticks(base_positions, datasets)
plt.legend(handles=legend_patches, loc='lower right')

# Save the plot as a high-resolution PNG file
plt.savefig('/content/Box-plot.pdf', dpi=300)


# Displaying the plot
plt.tight_layout()
plt.show()
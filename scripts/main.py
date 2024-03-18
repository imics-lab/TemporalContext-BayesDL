from load_data import get_dataset
from BayesMethod import learn_cpts, Bayesian_probabilities, combine_probabilities
from utils import tune_lambda_value
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt # for plotting training curves
import seaborn as sns

ds_list = [#"UniMiB SHAR",
           #"UCI HAR",
           "TWristAR",
           #"Leotta_2021",
           #"Gesture Phase Segmentation"
           ]
for i in ds_list:
    dataset = i
    print("**** ", dataset, " ****")

x_train, y_train, x_valid, y_valid, x_test, y_test, k_size, EPOCHS, t_names = get_dataset(dataset)
y = np.argmax(y_train, axis=-1) 
k = 20  # Number of previous states to consider
cpts = learn_cpts(y, k)  # Learning CPTs
with open('cpts.pickle', 'wb') as handle: 
    pickle.dump(cpts, handle, protocol=pickle.HIGHEST_PROTOCOL)   #save the CPTs from the training phase and use them later in the inference phase

with open('cpts.pickle', 'rb') as handle:
    cpts = pickle.load(handle)
num_classes = y_train.shape[1]  # Number of classes
sequence = y_test
loaded_probabilities = {}
for dataset in ds_list:
    loaded_probabilities[dataset] = np.load(f'predicted_probabilities_{dataset}.npy')
dl_probs_valid = loaded_probabilities[dataset] # Deep learning probabilities
dl_probs_test = loaded_probabilities[dataset] # Deep learning probabilities
lambda_values = np.linspace(0, 1, 11)  # Example list of lambda values
lambda_value = tune_lambda_value(x_valid, y_valid, cpts, dl_probs_valid, lambda_values)
bayesian_probs = Bayesian_probabilities(cpts, sequence, num_classes) # Calculating Bayesian probabilities
combined_probs = combine_probabilities(dl_probs_test, bayesian_probs, lambda_value) # Combining probabilities
new_y_pred = np.argmax(combined_probs, axis=-1)

print('Prediction accuracy: {0:.3f}'.format(accuracy_score(y_test, new_y_pred)))

# Print a report of classification performance metrics
print(classification_report(y_test, new_y_pred, target_names=t_names))

# Plot a confusion matrix
cm = confusion_matrix(y_test, new_y_pred)
cm_df = pd.DataFrame(cm,
                     index = t_names,
                     columns = t_names)
fig = plt.figure(figsize=(6.5,5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='cubehelix_r')
plt.title('CNN-Bayesian using '+dataset+'\nAccuracy:{0:.3f}'.format(accuracy_score(y_test, new_y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout() # keeps labels from being cutoff when saving as pdf
plt.show()
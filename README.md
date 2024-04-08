# TemporalContext-BayesDL
Enhancing Time-Series Prediction with Temporal Context Modeling: A Bayesian and Deep Learning Synergy

[![Conference](https://img.shields.io/badge/Conference-2024-4b44ce.svg)](https://www.flairs-37.info/home)

## Table of Contents
- [Description](#description)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Example](#Example)
- [Contributing](#contributing)
- [Citation](#citation)

## Description
This repository contains Python code using a combination of Deep Learning and Bayesian Models. Alongside training a deep learning model, we construct a Conditional Probability Table (CPT) during training to capture label transitions. During inference, these CPTs are utilized to adjust the predicted class probabilities of each window, taking into account the predictions of preceding windows. Our experimental analysis, focused on Human Activity Recognition (HAR) time series datasets, demonstrates that this approach not only surpasses the baseline performance of standalone deep learning models but also outperforms contemporary state-of-the-art methods that integrate temporal context into time series prediction.


## Dependencies
- Python 3.10
- TensorFlow 2.x
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn
- tsai
- PyTorch

## Installation

To install and run Temporal Context in Bayesian Deep Learning, follow these steps:

```bash
git clone https://github.com/imics-lab/TemporalContext-BayesDL.git
cd TemporalContext-BayesDL
conda env create -f environment.yml

```

## Usage

To use this project, run the main script after installation:

```bash
python scripts/main.py

```

## Models
  - [Baseline CNN](https://github.com/imics-lab/TemporalContext-BayesDL/blob/main/scripts/Baseline_CNN.ipynb)
  - [BayesDL](https://github.com/imics-lab/TemporalContext-BayesDL/blob/main/scripts/Bayesian_Method.ipynb) 
  - [InceptionTimePlus](https://github.com/timeseriesAI/tsai)
  - [TSTPlus](https://github.com/timeseriesAI/tsai)
  - [LSTNet](https://github.com/laiguokun/LSTNet)

## Example
Training:

```python 
from load_data import get_dataset
from BayesMethod import learn_cpts

x_train, y_train, x_valid, y_valid, x_test, y_test, k_size, EPOCHS, t_names = get_dataset(dataset)
y = np.argmax(y_train, axis=-1) 
k = 20  # Number of previous states to consider
cpts = learn_cpts(y, k)  # Learning CPTs
with open('cpts.pickle', 'wb') as handle: 
    pickle.dump(cpts, handle, protocol=pickle.HIGHEST_PROTOCOL)   #save the CPTs from the training phase and use them later in the inference phase
  ```
Inference:

```python
from BayesMethod import Bayesian_probabilities, combine_probabilities
from utils import tune_lambda_value

with open('cpts.pickle', 'rb') as handle:
    cpts = pickle.load(handle)
num_classes = y_train.shape[1]  # Number of classes
sequence = y_test
dl_probs = loaded_probabilities[dataset] # Deep learning probabilities
lambda_values = np.linspace(0, 1, 11)  # Example list of lambda values
lambda_value = tune_lambda_value(x_valid, y_valid, cpts, dl_probs_valid, lambda_values)
bayesian_probs = Bayesian_probabilities(cpts, sequence, num_classes) # Calculating Bayesian probabilities
combined_probs = combine_probabilities(dl_probs, bayesian_probs, lambda_value) # Combining probabilities
  ```

<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

<!-- CITATION -->
## Citation


```bibtex
@article{TemporalContext-BayesDL,
  title={Enhancing Time-Series Prediction with Temporal Context Modeling: 
         A Bayesian and Deep Learning Synergy},
  author={Irani, Habib and Metsis, Vangelis},
  booktitle={The International FLAIRS Conference Proceedings},
  year={2024}
}
```

# TemporalContext-BayesDL
Enhancing Time-Series Prediction with Temporal Context Modeling: A Bayesian and Deep Learning Synergy

## Table of Contents
- [Description](#description)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)
- [Citation](#citation)

## Description
This repository contains Python code using a combination of Deep Learning and Bayesian Models. Alongside training a deep learning model, we construct a Conditional Probability Table (CPT) during training to capture label transitions. During inference, these CPTs are utilized to adjust the predicted class probabilities of each window, taking into account the predictions of preceding windows. Our experimental analysis, focused on Human Activity Recognitio (HAR) time series datasets, demonstrates that this approach not only surpasses the baseline performance of standalone deep learning models but also outperforms contemporary state-of-the-art methods that integrate temporal context into time series prediction.


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

To set up the conda environment for this project, follow these steps:

1. Open a terminal.
2. Navigate to the project directory.
3. Run the following command to create the conda environment:

```bash
conda env create -f environment.yml

```

## Usage

 Clone the repository:

```bash
git clone https://github.com/imics-lab/TemporalContext-BayesDL.git
```

## Models
  - Baseline CNN 
  - BayesDL 
  - InceptionTimePlus 
  - LSTNet 

<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

<!-- CITATION -->
## Citation


```bibtex
@article{Habib Irani,
  title={Enhancing Time-Series Prediction with Temporal Context Modeling:
         A Bayesian and Deep Learning Synergy},
  author={(IMICS Lab) - Texas State University},
  year={2024}
}
```
# MTSST: Multivariate Time Series Step Tree

This repository contains the implementation of the **Multivariate Time Series Step Tree (MTSST)**, a novel interpretable machine learning model for multivariate time series classification.

## Project Overview

MTSST learns a decision tree where each node represents a "step" or subsequence pattern (shapelet) in the time series. It uses a custom distance metric based on FastDTW to identify discriminative patterns across multiple dimensions.

### Key Features
- **Interpretable**: The decision path clearly shows which time series patterns (witnesses) led to a classification.
- **Multivariate Support**: Handles multiple channels/dimensions natively.
- **Flexible**: Supports both Exclusive (search until match) and Non-Exclusive (search all) step learning strategies.

## Repository Structure

- `MTSST_Multivariate_Tree.py`: The main Python script containing the model implementation, training, and evaluation logic.
- `MTSST_Multivariate_Tree.ipynb`: The original Jupyter Notebook (requires updates to data paths if used directly).
- `Multivariate_ts_dataset/`: Directory containing time series datasets (e.g., Epilepsy).
- `requirements.txt`: List of Python dependencies.

## Installation

1.  Clone this repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: You need `graphviz` installed on your system for tree visualization.
- On macOS: `brew install graphviz`
- On Ubuntu: `sudo apt-get install graphviz`

## Usage

To run the model on the default **Epilepsy** dataset:

```bash
python MTSST_Multivariate_Tree.py
```

This script will:
1.  Load the train/test splits from `Multivariate_ts_dataset/Epilepsy/`.
2.  Train the MTSST model.
3.  Evaluate accuracy and print the classification report.
4.  Generate a visualization of the decision tree (`tree_exclusive_Epilepsy.png`).

### Using Custom Datasets

1.  Place your dataset in the `Multivariate_ts_dataset` folder.
2.  Ensure files are in the `.ts` format (sktime/aeon compatible) or update the `load_ts_file` function in the script.
3.  Modify the `dataset_name`, `train_file`, and `test_file` variables in `MTSST_Multivariate_Tree.py`.

## Citation

If you use this code in your research, please cite our paper:

[Insert Citation Here when available]

## License

[Insert License Here]

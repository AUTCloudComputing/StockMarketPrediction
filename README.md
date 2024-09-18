# StockMarketPrediction

## Setup

Switch to anaconda3 as current interpreter:

To get started:

### 1. **Install `cmake`**

To install it first, do this via `conda` or `apt`:

```bash
conda create -n myenv python=3.12
conda activate myenv
conda install numpy pandas scikit-learn
pip install -r requirements.txt
# pip install --ignore-installed -r requirements.txt
# Install cmake via conda
conda install -c conda-forge cmake

# Alternatively, you can install it via apt
sudo apt-get install cmake
```

### 2. **Install other dependencies**
After installing `cmake`, it’s also good to ensure that other build tools are available, such as `pkg-config` and `build-essential`:

```bash
# Install build-essential and pkg-config via apt
sudo apt-get install build-essential pkg-config
```

### 3. **Try installing `sentencepiece` separately**
Now that you have the necessary tools, try installing `sentencepiece` separately to see if it builds properly:

```bash
pip install sentencepiece
```

### 4. **Re-run `pip install -r requirements.txt`**
Once `sentencepiece` is successfully installed, re-run the `requirements.txt` installation:

```bash
pip install -r requirements.txt
```

All setup steps:

```bash
conda create -n sep_env python=3.12

conda activate sep_env

conda install python=3.12  # if using conda

## remove '#torch==2.2.0+cu118' from requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install cmake via conda
conda install -c conda-forge cmake

# Alternatively, you can install it via apt
sudo apt-get install cmake

# Install build-essential and pkg-config via apt
sudo apt-get install build-essential pkg-config

pip install sentencepiece

pip install -r requirements.txt
```

set openai API key:

```bash
vim ~/.bashrc 
export OPENAI_API_KEY="your_actual_api_key"
source ~/.bashrc 
echo $OPENAI_API_KEY
```

setup hugging face

```bash

pip install huggingface_hub==0.24.6
pip install tokenizers==0.14.1
pip install --upgrade transformers
pip install --upgrade trl 

```

Run a sample experiment:

```bash

python main.py --price_dir "data/sample_price/preprocessed/" --tweet_dir "data/sample_tweet/raw/"

```


## How it works

###  Scikit-Learn 

To prevent overfitting during the model training process, we implemented Train-Test Evaluation using Scikit-Learn. In this setup, 80% of the data is allocated for training, while 20% is reserved for testing. This approach helps ensure that the model generalizes well to unseen data.

To prevent overfitting during the model training process, we implemented Train-Test Evaluation using Scikit-Learn. In this setup, 80% of the data is allocated for training, while 20% is reserved for testing. This method ensures that our model generalizes well to unseen data and doesn't just memorize the training data.

In our dataset preparation, we load and structure the data using the `DataLoader` class. This class handles the stock price and related tweet data, which we use for both training and testing the model's ability to predict stock price movements based on historical data.

### Dataset Splitting Process

1. **Training Dataset (80% of the data)**:
    - The first 80% of the stock price data is designated for training the model. This portion allows the model to learn from historical stock prices and tweet data. Specifically, we fetch stock prices, generate tweet summaries, and determine sentiment associated with those prices.
    - The following code snippet outlines the dataset splitting process:

      ```python
      tes_idx = round(len(ordered_price_data) * 0.8)
      if flag == "train":
          data_range = range(tes_idx)  # First 80% for training
      ```

2. **Test Dataset (20% of the data)**:
    - The remaining 20% of the stock price data is reserved for testing. This test set ensures that we can measure the model's performance on unseen data and evaluate whether it has successfully generalized.

      ```python
      if flag != "train":
          data_range = range(tes_idx, end_idx)  # Last 20% for testing
      ```

By clearly separating the training and test datasets, we ensure that the model is evaluated on fresh data that it has not encountered during training. This is a crucial step in validating the model's predictive power and avoiding overfitting.

### Code Example for Dataset Preparation

The following code demonstrates how we split the dataset into training and testing sets:

```python
tes_idx = round(len(ordered_price_data) * 0.8)
train_data_range = range(tes_idx)     # First 80% for training
test_data_range = range(tes_idx, len(ordered_price_data))  # Last 20% for testing

for idx in train_data_range:
    # Training dataset preparation
    # Fetch stock prices, get tweet summaries, and calculate sentiment
    ...

for idx in test_data_range:
    # Test dataset preparation
    # Fetch stock prices, get tweet summaries, and calculate sentiment
    ...
```

In our approach, the **training dataset** helps the model fine-tune its understanding of stock market movements, while the **test dataset** provides a benchmark to assess its performance and generalization capabilities. Through this method, we can confidently train a robust model while minimizing the risk of overfitting.
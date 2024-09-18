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
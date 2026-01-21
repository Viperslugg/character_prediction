# character_prediction
This repository is to implement, train and optimize a basic transformer model to predict the next character after encountering a sequence of previous characters.

This project follows a sequence of methods and techniques to optimize a basic transformer model by adjusting model architecture, training procedure and hyperparameters. It follows from **transformer_I.ipynb** and ends at **transformer_VI.ipynb**. The breakdown of experiments is summarized below:
- transformer_I: Initial tuning of model parameters to start with an initial basic transformer model
- transformer_II: Tuning of multi-layer perceptron (MLP) ratio, dropout and experimentation with sinusoidal positional encodings (P.E)
- transformer_III: Different activation functions, batch size, learning rate and type of optimizer (Adam, AdamW and Lion)
- transformer_IV: Weight decay, gradient clipping and learning rate scheduling strategies (with warmup)
- transformer_V: Scaling (dataset and model sizes) and implementing a different loss function design to analyze if computing the averaged cross-entropy loss in another way is more effective in training the transformer
- transformer_VI: Optimization of the loss function, final training of model

The folder **conf** holds the configuration files:
- config1: Mlp ratio and dropout are varying, used in transformer_II.ipynb
- config2: Batch size and learning rate are varying, used in transformer_III.ipynb
- config3: Activation functions, batch size and learning rate are varying, used in transformer_III.ipynb
- config4: Weight decay and gradient clipping thresholds are varying, used in transformer_IV.ipynb

The folder **scripts** contains **functions.py**, a file that has some of the modules and functions needed to be used in all the Jupyter notebook (.ipynb) files.
The folder **models** contains the following files:
- basic_transformer.py: A basic transformer model used in transformer_I.ipynb
- modified_transformer.py and pos_encoding_transformer.py: Modified transformer models used in transformer_II.ipynb for addition of mlp ratio and dropout, and experimentation with sinusoidal P.E respectively
- transformer_III.py: the main transformer model used in transformer_III.ipynb and beyond. Adapted from modified_transformer.py and includes modification for experimenting with activation functions.

This code was ran on Google Colab; the code for importing of data/files must be amended to your preference. Configuration files may not neccessarily be needed for a small project like this, but is good because they separate concerns without manually editing the code, although I relied on them less as the project progresses due to time constraints.


Below is the structure of the repository:

```plaintext
    .
    ├── transformer_I.ipynb
    ├── transformer_II.ipynb
    ├── transformer_III.ipynb
    ├── transformer_IV.ipynb
    ├── transformer_V.ipynb
    ├── transformer_VI.ipynb
    ├── models/
    │   ├── basic_transformer.py
    │   ├── modified_transformer.py
    │   ├── pos_encoding_transformer.py
    │   └── transformer_III.py
    ├── scripts/
    │   └── functions.py
    ├── conf/
    │   ├── config1.yml
    │   ├── config2.yml
    │   ├── config3.yml
    │   └── config4.yml
    ├── README.md
    ├── test_text_int.npy
    ├── text8_test.txt
    ├── text8_train.txt
    └── train_text_int.npy

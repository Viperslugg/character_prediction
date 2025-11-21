# character_prediction
This repo is designed to implement and optimize a basic transformer model to predict the next character after encountering a sequence of previous characters.

The work done here was inspired by [this repository](https://github.com/alexxthiery/char_transformer). Special thanks to [Alexandre Thiery](https://alexxthiery.github.io/) for answering my questions in [this assignment](https://alexxthiery.github.io/teaching/character_LLM/charllm.html).


This project follows a sequence of methods to optimize a basic transformer model by adjusting model architecture, training procedure and hyperparameters. It follows from **transformer_I.ipynb** and ends at **transformer_VI.ipynb**. The breakdown of experiments is summarized below:
- transformer_I: Initial tuning of model parameters to get an intial model to start with
- transformer_II: Tuning of mlp ratio, dropout and experimentation with sinusoidal P.E
- transformer_III: Different activation functions, batch size, learning rate and optimizer
- transformer_IV: Weight decay, gradient clipping and learning rate scheduling strategies (with warmup)
- transformer_V: Scaling (dataset and model sizes), implementing a different loss function design
- transformer_VI: Optimization of L, final training of model

The folder **conf** holds the configuration files:
- config1: Mlp ratio and dropout are varying, used in transformer_II.ipynb
- config2: Batch size and learning rate are varying, used in transformer_III.ipynb
- config3: Activation functions, batch size and learning rate are varying, used in transformer_III.ipynb
- config4: Weight decay and gradient clipping thresholds are varying, used in transformer_IV.ipynb

The folder **scripts** contains **functions.py**, a file that has some of the modules and functions needed to be used in all the .ipynb files.
The folder **models** contains the following files:
- basic_transformer.py: A basic transformer model used in transformer_I.ipynb
- modified_transformer.py and pos_encoding_transformer.py: Modified transformer models used in transformer_II.ipynb for addition of mlp ratio and dropout, and experimentation with sinusoidal P.E respectively
- transformer_III.py: the main transformer model used in transformer_III.ipynb and beyond. Adapted from modified_transformer.py and includes modification for experimenting with activation functions.

This code was ran on Google Colab; the code for importing of data/files must be amended for your preference. Configuration files may not neccessarily be needed for a small project like this, but is good because they separate concerns and shape the project's structures, although I relied on them less as the project progresses.


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

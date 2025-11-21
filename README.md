# character_prediction
This repo is designed to implement and optimize a basic transformer model to predicti the next character after encountering a sequence of previous characters.

The work done here was inspired by [this repository](https://github.com/alexxthiery/char_transformer). Special thanks to [Alexandre Thiery](https://alexxthiery.github.io/) for answering my questions in [this assignment](https://alexxthiery.github.io/teaching/character_LLM/charllm.html).

Below is the structure of the repository

```plaintext
    .
    ├── src/
    │   ├── transformer_I.ipynb
    │   ├── transformer_II.ipynb
    │   ├── transformer_III.ipynb
    │   ├── transformer_IV.ipynb
    │   ├── transformer_V.ipynb
    │   ├── transformer_VI.ipynb
    │   └── mixed_precision.ipynb
    ├── models/
    │   ├── basic_transformer.py
    │   ├── modified_transformer.py
    │   ├── pos_encoding_transformer.py
    │   └── transformer_III.py
    ├── scripts/
    │   ├── generation.py
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
    ├── train_text_int.npy

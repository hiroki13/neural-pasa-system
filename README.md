# ニューラル述語項構造解析器

This repo contains Theano implementations of our original models and the models described in the following papers:

[Semi-supervised Question Retrieval with Gated Convolutions](http://arxiv.org/abs/1512.05726). NAACL 2016


## 日本語述語項構造解析

ガ格・ヲ格・ニ格を当てる．


### データセット
The data can be downloaded at this [repo](https://github.com/taolei87/askubuntu).

### パッケージ依存
To run the code, you need the following extra packages installed:
  - Numpy and Theano

#### 使用法
  1. Clone this repo
  2. Run `python -m sim_q_ranking.main.main --help` to see all running options

#### 使用コマンド例
  - Basic Model: `python -m sim_q_ranking.main.main --corpus path/to/data/text_tokenized.txt.gz --embeddings path/to/data/vector/vectors_pruned.200.txt.gz --train path/to/data/train_random.txt --dev path/to/data/dev.txt --test path/to/data/test.txt --layer rcnn`


# Demystifying Distributed Training of Graph Neural Networks for Link Prediction

## Overview

This repository contains the implementation of SpLPG, a distributed graph neural network (GNN) training framework for link prediction.


## Requirements

<!--PyTorch v2.0.1-->
<!--DGL=1.1.0-->
<!--CUDA=11.8-->

[![](https://img.shields.io/badge/PyTorch-2.0.1-blueviolet)](https://pytorch.org/get-started/)
[![](https://img.shields.io/badge/DGL-1.1.0-blue)](https://www.dgl.ai/pages/start.html)
[![](https://img.shields.io/badge/CUDA-11.8-green)](https://developer.nvidia.com/cuda-11-8-0-download-archive)

GPU version of [DGL](https://www.dgl.ai/pages/start.html) are required to run SpLPG. Please check its official website for installation.

## Installation

We recommend using the conda virtual environment

```bash
$ conda env create -f environment.yml
```

The installation of conda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Framework
![Image text](https://github.com/xhuang2016/SpLPG/blob/main/SpLPG.png)


## Dataset
We use 9 real-world graph datasets that are downloaded from [DGL](https://www.dgl.ai/) and [OGB](https://ogb.stanford.edu/).


## Running the code

Follow the command below to run the code.

1. Download and preprocess dataset
```bash
$ python3 preprocessing.py --arguments [xxx]
```

2. Partition the graph
```bash
$ python3 partitioning.py --arguments [xxx]
```

3. Training the GNN model
```bash
$ python3 file_name.py --arguments [xxx]
```

For example, run the following commands to train a GraphSAGE model via SpLPG with 4 workers on Cora dataset:
```bash
$ python3 preprocessing.py --dataset cora
$ python3 partitioning.py --dataset cora --number_partition 4 --method SpLPG
$ python3 SpLPG.py --dataset cora --number_partition 4 --model_name GraphSAGE
```

Note:
> - Please check each **.py file for more detailed arguments.
> - Please refer to our paper for more details.  

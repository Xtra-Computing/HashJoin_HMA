# HashJoin_HMA
A hash join implementation optimized for many-core processors with die-stacked HBMs.

## Overview

### Many-core and die-stacked HBM
Recently, die-stacked high bandwidth memory (HBM) has emerged in CPU architectures. The high bandwidth of the HBM makes the HBM a good partner for many-core processors which are usually bandwidth-hungry when coupled with only the off-package main memory. For example, the Intel KNL many-core CPU has equipped with a 16 GB HBM that achieves the peak memory bandwidth of about 400 GB/s and up to 72 x86-based CPU cores. 

### Motivation
In main-memory databases, hash tables are important data structures for query processing whose application are heavily influenced by the underlying memory. An efficient deployment of hash tables on the many-core processor with die-stacked HBMs is technically challenging because of the heterogeneity of hardware resources as well as the random access pattern on the hash table. The state-of-the-art deployment algorithms designed for multi-core processors assume a single type of memory with a consistent memory bandwidth across NUMA nodes. This assumption no longer holds for the die-stacked HBMs and many-core processors, and new performance optimizations must be developed for hash table deployment. 

### Contribution
In this implementation of hash joins, we propose and apply a deployment algorithm for hash tables which optimize the placement of hash tables between the die-stacked HBMs and main memory, and the scheduling of threads accessing the hash table on many-core architectures. We perform the experiments on the Intel KNL many-core CPU. Evaluation results show that our proposed deployment of hash table can achieve about two times performance improvement over the state-of-the-art hash join algorithms.

## Prerequisites

### Hardware

* Intel Xeon Phi Many-core processor of the Knights Landing Architecture 

### Software

* Linux 
* Intel C/C++ 17.0.2 20170213
* The [memkind library](https://github.com/memkind/memkind)

## Download

```bash
git clone https://github.com/PatrickXC/HashJoin_HMA.git
```

## Build

```bash
make hj_hma
```

## Generate inputs

```bash
make write
./write [size of outer relation] [size of inner relation] [number of threads]
```

## Run
```bash
./hj_hma [size of outer relation] [size of inner relation] [number of threads]
```

## Cite this work
If you use it in your paper, please cite our work ([full version](https://www.comp.nus.edu.sg/~hebs/pub/hbm-cikm19.pdf)).
```
@inproceedings{10.1145/3357384.3358015,
author = {Cheng, Xuntao and He, Bingsheng and Lo, Eric and Wang, Wei and Lu, Shengliang and Chen, Xinyu},
title = {Deploying Hash Tables on Die-Stacked High Bandwidth Memory},
year = {2019},
isbn = {9781450369763},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3357384.3358015},
doi = {10.1145/3357384.3358015},
booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
pages = {239–248},
numpages = {10},
keywords = {die-stacked high bandwidth memory, hash joins},
location = {Beijing, China},
series = {CIKM ’19}
}

```
### Related publications
* Saurabh Jha, Bingsheng He, Mian Lu, Xuntao Cheng, and Huynh Phung Huynh. 2015. [Improving main memory hash joins on Intel Xeon Phi processors: an experimental approach](http://www.vldb.org/pvldb/vol8/p642-Jha.pdf). Proc. VLDB Endow. 8, 6 (February 2015), 642–653.

* Xuntao Cheng, Bingsheng He, Xiaoli Du, and Chiew Tong Lau. 2017. [A Study of Main-Memory Hash Joins on Many-core Processor: A Case with Intel Knights Landing Architecture](https://www.comp.nus.edu.sg/~hebs/pub/Hash_Join_on_KNL_CIKM17.pdf). In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM ’17). Association for Computing Machinery, New York, NY, USA, 657–666. 

* Xuntao Cheng, Bingsheng He, Mian Lu, Chiew Tong Lau, Huynh Phung Huynh, and Rick Siow Mong Goh. 2016. [Efficient Query Processing on Many-core Architectures: A Case Study with Intel Xeon Phi Processor](https://dl.acm.org/doi/10.1145/2882903.2899407). In Proceedings of the 2016 International Conference on Management of Data (SIGMOD ’16). Association for Computing Machinery, New York, NY, USA, 2081–2084. 

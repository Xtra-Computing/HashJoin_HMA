# HashJoin_HMA
A hash join implementation optimized for many-core processors with die-stacked HBMs.

## Overview

Recently, die-stacked high bandwidth memory (HBM) has emerged in CPU architectures. The high bandwidth of the HBM makes the HBM a good partner for many-core processors which are usually bandwidth-hungry when coupled with only the off-package main memory. For example, the Intel KNL many-core CPU has equipped with a 16 GB HBM that achieves the peak memory bandwidth of about 400 GB/s and up to 72 x86-based CPU cores. In main-memory databases, hash tables are important data structures for query processing whose application are heavily influenced by the underlying memory. An efficient deployment of hash tables on the many-core processor with die-stacked HBMs is technically challenging because of the heterogeneity of hardware resources as well as the random access pattern on the hash table. The state-of-the-art deployment algorithms designed for multi-core processors assume a single type of memory with a consistent memory bandwidth across NUMA nodes. This assumption no longer holds for the die-stacked HBMs and many-core processors, and new performance optimizations must be developed for hash table deployment. 

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

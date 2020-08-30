# Inductive-Hypergraph-Neural-Network-based-on-Hypergraph-Sampler
The repo implement a Inductive Hypergraph Neural Network(Inductive-HGNN) with Hypergraph Neighbors Sampler(HySampler), and perform this on OGB-products.

We implement Hypergraph Neighbors Sampler by C++ instead of Python to improve the performance, there are almost **50 times** improvement, 
which is necessary for train in large scale dataset, e.g. ogb-products.

Although the Inductive-HGNN do not have a comparable performance for ogb-products(maybe some improvements can be made for HGNN), 
we think the faster HySampler is userful for the researchers who are interested in HGNN.

Unfortunately I ended the internship in iMoon-lab and didn't have computing resources(GPUs) to do more following works. 
If you are interested in this work, please contact my Email(djytyang@Gmail.com) for cooperation.


## Installation and Run 
### Dependency
* PyTorch
* PyTorch-Geometric
* THU-DeepHypergraph
* tqdm

### Install C++ version Hypergraph Neighbor Sampler 
```shell script
cd hysample_cpp
python setup.py install 
```

### Run 
```shell script
python cora_main.py
python ogb_main.py
```
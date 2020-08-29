# Inductive-Hypergraph-Neural-Network-based-on-Hypergraph-Sampler
The repo implement a Inductive Hypergraph Neural Network(Inductive-HGNN) with Hypergraph Neighbors Sampler(HySampler), and perform this on OGB-products.

We implement Hypergraph Neighbors Sampler by C++ instead of Python to improve the performance, there are almost **50 times** imporvement, 
which is neccessary for train in large scale dataset, e.g. ogb-products.

Although the Inductive-HGNN do not have a comparable performance for ogb-products(maybe some improvements can be made for HGNN), 
we think the faster HySampler is userful for the researchers who are interested in HGNN.

Unfortunately I ended the internship in iMoon-lab@THU and didn't have computing resources(GPUs) to do more following works. 
If you are interested in this work, please contact my Email(djytyang@Gmail.com) for cooperation.

We will release the code in few days, thanks for your attention.

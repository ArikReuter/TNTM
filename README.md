# TNTM

This repository contains the code for the **T**ransformer-Representation **N**eural **T**opic **M**odel (TNTM) based on the paper "Probabilistic Topic Modelling with Transformer
Representations" by Arik Reuter, Anton Thielmann, Christoph Weisser, Benjamin Säfken and Thomas Kneib. 
Our model combines the benefits of topic representations in transformer-based embedding spaces and probabilistic modelling as in, for example, Latent Dirichlet Allocation. 
Therefore, this approach unifies the powerful and versatile notion of topics based on transformer embeddings with probabilistic modelling. 
TNTM is able to find coherent and diverse topics in large datasets and is able to extract large numbers of topics without degrading performance. 



### Preprocessing

The folder Data/ contains an example notebook for obtaining the 20 Newsgroups corpus (https://github.com/MIND-Lab/OCTIS) and the Reuters corpus (https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection).

In Code/Preprocessing, we inlcuded two Jupyter notebooks exemplifying the potential further processing of those two corpora in order to obtain BERT-based word embeddings. 

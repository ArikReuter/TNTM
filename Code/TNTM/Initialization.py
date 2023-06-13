# -*- coding: utf-8 -*-
import torch
from torch import nn
#import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as nnf
import collections
from collections import namedtuple
from tqdm import tqdm

from sklearn.datasets import fetch_20newsgroups
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import umap

from sklearn.mixture import GaussianMixture


from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device: {device}')


class Initializer:
  def __init__(self, embeddings_vocab, n_topics, n_dims = 11):
    """
    Args: 
      embeddings_vocab: embedding of each word in the vocabulary 
      n_topics: number of topics to find
      n_dims: Number of dimensions to project data into with UMAP 
    """
    self.embeddings_vocab = embeddings_vocab
    self.n_topics = n_topics
    self.n_dims = n_dims 

  def reduce_dimensionality(self, umap_hyperparams = {'n_neighbors': 15, 'min_dist': 0.01}):
    """
    Take the tensor embeddings of the embeddings of the vocabulary and reduce their dimensionality. 
    The paramters n_neighbors and min_dist change the behavour of UMAP. 
    """
    
    umap1 = umap.UMAP(n_components=self.n_dims, metric = 'cosine', **umap_hyperparams)
    proj_embeddings = umap1.fit_transform(self.embeddings_vocab)
    self.proj_embeddings = proj_embeddings

    return proj_embeddings

  def fit_gmm(self, embeddings, random_state = 42):
    """
    Fit Gaussian Mixture model to the embeddings (with dimensionality reduction)
    and return the menas and covariances of the Gaussians and the bic of this model
    """

    if embeddings is not None: 
      embeddings = self.proj_embeddings


    # fit gmm to embeddings 
    gmm1 = GaussianMixture(n_components=self.n_topics,covariance_type='full', random_state = random_state)
    gmm1.fit(embeddings)

    mus_init = torch.tensor(gmm1.means_)
    sigmas_init = torch.tensor(gmm1.covariances_)

    bic = gmm1.score(embeddings)

    return mus_init, sigmas_init, bic

  def get_reparametrization_parameters(self, sigmas_init, eps = 1e-4):
    """
    Compute the paramters for the reparamterization of the sigmas, such that \sigma = L_lower_init @ L_lower_init.T + torch.exp(log_diag_init) where log_diag_init is a diagonal matrix 
    """

    L_lower_init = torch.linalg.cholesky(sigmas_init)
    log_diag_init = torch.log(torch.ones(self.n_topics, self.n_dims)*1e-4)  # initialize diag = (1,...,1)*eps, such that only a small value is added to the diagonal

    return L_lower_init, log_diag_init


  def reduce_dim_and_cluster(self, eps = 1e-4, umap_hyperparams = {'n_neighbors': 15, 'min_dist': 0.01}):
    """
    Reduce the dimensionality with UMAP of the embeddings and fit a GMM model, which yields the means and covariances (albeit reparameterized) 
    of the GMM. 
    Args: 
      n_neigbors: Number of neighbors to consider in UMAP
      min_dist: Minimal distance of points in space with lower dimensionality for UMAP
      eps: Regularization paramter for the covariance matrices. 

    Return: 
      mus_init: means of topic-specific covariances 
      L_lower_init: factor of covariance matrix
      log_diag_init: log of diagonal matrix to add to L_lower_init @ L_lower_init.T 
      bic: Bayesian information criterion of GMM
    """
  
    emb_dim_red = self.reduce_dimensionality(umap_hyperparams = umap_hyperparams)

    mus_init, sigmas_init, bic = self.fit_gmm(emb_dim_red)

    L_lower_init, log_diag_init = self.get_reparametrization_parameters(sigmas_init, eps = eps)

    return emb_dim_red, mus_init, L_lower_init, log_diag_init, bic


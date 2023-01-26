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
import time



from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device: {device}')




class Linear_skip_block(nn.Module):
  """
  Block of linear layer + softplus + skip connection +  dropout  + batchnorm 
  """
  def __init__(self, n_input, dropout_rate):
    super(Linear_skip_block, self).__init__()

    self.fc = nn.Linear(n_input, n_input)
    self.act = torch.nn.LeakyReLU()

    self.bn = nn.BatchNorm1d(n_input, affine = True) 
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    x0 = x
    x = self.fc(x)
    x = self.act(x)
    x = x0 + x
    x = self.drop(x)
    x = self.bn(x)

    return x


class Linear_block(nn.Module):
  """
  Block of linear layer dropout  + batchnorm 
  """
  def __init__(self, n_input, n_output, dropout_rate):
    super(Linear_block, self).__init__()

    self.fc = nn.Linear(n_input, n_output)
    self.act = torch.nn.LeakyReLU()
    self.bn = nn.BatchNorm1d(n_output, affine = True) 
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    x = self.fc(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.bn(x)

    return x


class Encoder_NVLDA(nn.Module):
  """
  Encoder for TLDA, takes tokenized bow representation of a batch of documents and returns the mean and log-variance of the corresponding distributions over theta 
  """
  def __init__(self, config):
    super(Encoder_NVLDA, self).__init__()

    self.config = config

    self.linear1 = Linear_block(config.num_input, config.n_hidden_block, config.drop_rate_en)    # initial linear layer 
    self.hidden_layers = torch.nn.Sequential(*[Linear_skip_block(config.n_hidden_block, config.drop_rate_en) for _ in range(config.n_skip_layers)])  #hidden skip-layers
    self.mean_fc = nn.Linear(config.n_hidden_block, config.n_topics)              #linear layer to get mean of topic-distribution per document
    self.logvar_fc = nn.Linear(config.n_hidden_block, config.n_topics)            #linear layer to get log of diagonal of covariance of topic-distribution per document
    self.act = nnf.softplus                                                       #softplus activation function

  def forward(self, x):
    x = self.linear1(x)
    x = self.hidden_layers(x)
  
    posterior_mean = self.mean_fc(x)                                            #calculate posterior mean from output of second linear layer
    posterior_logvar = self.logvar_fc(x)                                        # calculate posterior logvar from output of seconf linear layer

    return posterior_mean, posterior_logvar

class Encoder_SentenceTransformer(nn.Module):
  """
  Encoder for TLDA, takes tokenized bow representation of a batch of documents and returns the mean and log-variance of the corresponding distributions over theta 
  """
  def __init__(self, config, sentence_transformer, extract_from_sentence_transformer):
    """
    Use a sentence transformer in the encoder, as in Bianchi et al. 2021.
    Args: 
        config: config file
        sentence_transformer: sentence transformer to use in the encoder
        extract_from_sentence_transformer: function that maps the output of the sentence transformer into something useable 
        
    """
    super(Encoder_NVLDA, self).__init__()

    self.config = config
    self.sentence_transformer = sentence_transformer
    
    self.linear_layer = nn.Linear(config.sentence_transformer.hidden_dim, config.n_hidden_block) #linear layer from sentence transformer to mean and logvar layers    
    self.mean_fc = nn.Linear(config.n_hidden_block, config.n_topics)              #linear layer to get mean of topic-distribution per document
    self.logvar_fc = nn.Linear(config.n_hidden_block, config.n_topics)            #linear layer to get log of diagonal of covariance of topic-distribution per document
    
    self.act = nnf.softplus 


  def forward(self, x):
    """
    Takes batches of sentences as input and outputs the variational posterior mean and variational posterior logvariance to sample from. 
    """
    x = self.sentence_transformer(x)
    x = self.linear_layer(x)
    x = self.act(x)
  
    posterior_mean = self.mean_fc(x)                                            #calculate posterior mean from output of second linear layer
    posterior_logvar = self.logvar_fc(x)                                        # calculate posterior logvar from output of seconf linear layer

    return posterior_mean, posterior_logvar
    
    
class Encoder_SentenceTransformer_precomputed(nn.Module):
  """
  Encoder for TLDA, takes tokenized bow representation of a batch of documents and returns the mean and log-variance of the corresponding distributions over theta 
  """
  def __init__(self, config):
    """
    Use the embedding obtained by a sentence transformer
    Args: 
        config: config file
        sentence_transformer: sentence transformer to use in the encoder
        extract_from_sentence_transformer: function that maps the output of the sentence transformer into something useable 
        
    """
    super(Encoder_SentenceTransformer_precomputed, self).__init__()

    self.config = config
  
    self.linear_layer = nn.Linear(config.sentence_transformer_hidden_dim, config.n_hidden_block) #linear layer from sentence transformer to mean and logvar layers    
    self.mean_fc = nn.Linear(config.n_hidden_block, config.n_topics)              #linear layer to get mean of topic-distribution per document
    self.logvar_fc = nn.Linear(config.n_hidden_block, config.n_topics)            #linear layer to get log of diagonal of covariance of topic-distribution per document
    
    self.act = nnf.softplus 


  def forward(self, x):
    """
    Takes batches of embeddings of sentences as input and outputs the variational posterior mean and variational posterior logvariance to sample from. 
    """
    x = self.linear_layer(x)
    x = self.act(x)
  
    posterior_mean = self.mean_fc(x)                                            #calculate posterior mean from output of second linear layer
    posterior_logvar = self.logvar_fc(x)                                        # calculate posterior logvar from output of seconf linear layer

    return posterior_mean, posterior_logvar
    
    


def calc_beta(mus, L_lower, log_diag, embeddings, config):
  """
  take parameters of topic-specific normal distributions of shape (n_topics, embedding_dim), i.e. mus and L_lower
  and return probability of each word embedding among the embeddings. 
  L_lower is a (n_embedding_dim, n_embedding_dim) matrix, but only the part below the diagonal is used 

  Return log-probabilities of each embedding under each normal distribution 
  """

  diag = torch.exp(log_diag)

  normal_dis_lis = [LowRankMultivariateNormal(mu, cov_factor= lower, cov_diag = D) for mu, lower, D in zip(mus, L_lower, diag)]
  log_probs = torch.zeros(config.n_topics, config.vocab_size)

  for i, dis in enumerate(normal_dis_lis):                                     # possible speedup with vamp?                   
    log_probs[i] = dis.log_prob(embeddings)
  
  return log_probs


class Decoder_TNTM(nn.Module):
  """
    embeddings: The embeddings of every word in the corpus
    mus_init: What to initialize the means with 
    L_lower_init: What to initialize the L matrix for the variance with 
    log_diag_init: What to initialize the log of the diagonal component of the variance with. The covariance is reparametrized as sigma = L_lower_init.T @ L_lower_init + exp(log_diag_init)
    config: config dict 
    """
  def __init__(self, embeddings, mus_init, L_lower_init, log_diag_init, config):
    """
    embeddings: The precomputed embeddings of every word in the corpus
    mus_init: What to initialize the means with 
    L_lower_init: What to initialize the 
    """
    super(Decoder_TNTM, self).__init__()
    
    self.config = config
    self.emebddings = embeddings


    self.mus = nn.Parameter(mus_init)   #create topic means as learnable paramter
    self.L_lower = nn.Parameter(L_lower_init)   # factor of covariance per topic
    self.log_diag = nn.Parameter(log_diag_init)  # summand for diagonal of covariance 
    self.embeddings = embeddings

  def forward(self, theta_hat):
    """
    theta_hat is directly sampled according to theta_hat ~ mu_0 + \Sigma_0^{1/2}@\epsilon for \epsilon ~ N(0, Id)
    """
    batch_size = theta_hat.shape[0]

    log_beta = calc_beta(self.mus, self.L_lower, self.log_diag, self.emebddings, self.config).to(device)   #calculate log_beta, i.e. log-likelihoods of each embedding under the current topics 

    beta_batched_shape = (batch_size, ) + log_beta.shape
    log_beta = log_beta.expand(*beta_batched_shape)    


    # use numerical trick to compute log(beta @ theta )
    log_theta = torch.nn.LogSoftmax(dim=-1)(theta_hat)                           #calculate log theta = log_softmax(theta_hat)
    A = log_beta + log_theta.unsqueeze(-1)                                       # calculate (log (beta @ theta))[i] = (log (exp(log_beta) @ exp(log_theta)))[i] = log(\sum_k exp (log_beta[i,k] + log_theta[k]))
    x = torch.logsumexp(A, dim = 1)                            
    return x


class TNTM_bow(nn.Module):
  """
  Combine encoder and decoder to one model 
  """

  def __init__(self, config, embeddings, mus_init, lower_init, log_diag_init, prior_mean, prior_variance):
    super(TNTM_bow, self).__init__()
    
    self.config = config

    self.encoder = Encoder_NVLDA(config)                                        #use same encoder as for NVLDA
    self.decoder = Decoder_TNTM(embeddings, mus_init, lower_init, log_diag_init, config)  # Use decoder from TGLDA 
    
    self.prior_mean = prior_mean
    self.prior_variance = prior_variance


  def forward(self, x):
    posterior_mean, posterior_logvar = self.encoder(x)                          # encoder
    posterior_std = torch.exp(0.5*posterior_logvar)   

    eps = torch.randn_like(posterior_std)                                       #random sample of eps ~ (N(0,1), ..., N(0,1))^T Does not compute gradient by default
                                     
    theta_hat = posterior_mean + eps*posterior_std                              #sample theta_hat from the logistic normal distribution 


    log_recon = self.decoder(theta_hat)                                         #use decoder to reconstruct the result, i.e. multiply beta @ theta 

    return log_recon, posterior_mean, posterior_logvar                          # return reconstruction loss and paramters of variational posterior of theta|document for the ELBO 
    

class TNTM_sentence_transformer_precomputed(nn.Module):
  """
  Combine encoder and decoder to one model 
  """

  def __init__(self, config, embeddings, mus_init, lower_init, log_diag_init, prior_mean, prior_variance):
    super(TNTM_sentence_transformer_precomputed, self).__init__()
    
    self.config = config

    self.encoder = Encoder_SentenceTransformer_precomputed(config)                                        #use same encoder as for NVLDA
    self.decoder = Decoder_TNTM(embeddings, mus_init, lower_init, log_diag_init, config)  # Use decoder from TGLDA 
    
    self.prior_mean = prior_mean
    self.prior_variance = prior_variance


  def forward(self, x):
    posterior_mean, posterior_logvar = self.encoder(x)                          # encoder
    posterior_std = torch.exp(0.5*posterior_logvar)   

    eps = torch.randn_like(posterior_std)                                       #random sample of eps ~ (N(0,1), ..., N(0,1))^T Does not compute gradient by default
                                     
    theta_hat = posterior_mean + eps*posterior_std                              #sample theta_hat from the logistic normal distribution 


    log_recon = self.decoder(theta_hat)                                         #use decoder to reconstruct the result, i.e. multiply beta @ theta 

    return log_recon, posterior_mean, posterior_logvar                          # return reconstruction loss and paramters of variational posterior of theta|document for the ELBO 


def loss_elbo(input, log_recon, posterior_mean, posterior_logvar, prior_mean, prior_var, n_topics, reg = 1e-10):
  """
  Calculate the elbo for bow input u^d, the log-likelihood of reconstructing individual words, the posterior mean and posterior logvar for the KLD, 
  the paramters prior_mean and prior_var also for the KLD, the number of topics and a regularization paramter 
  """

  #Negative log-likelihood:  - (u^d)^T @ log(beta @ \theta^d)
  NL = -(input * log_recon).sum(1)  
  
  #KLD between variational posterior p(\theta|d) and prior p(\theta)
  posterior_var = posterior_logvar.exp()
  prior_mean = prior_mean.expand_as(posterior_mean)  
  prior_var = prior_var.expand_as(posterior_mean)
  prior_logvar = torch.log(prior_var)
  

  var_division = posterior_var / prior_var


  diff = posterior_mean - prior_mean
  diff_term = diff*diff / prior_var
  logvar_division = prior_logvar - posterior_logvar


  KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - n_topics)

  NL_avg = torch.mean(NL)   #average loss over entire batch
  KLD_avg = torch.mean(KLD)

  loss = (NL + KLD).mean()

  return loss, NL_avg, KLD_avg
  
  
  
def train_test_split(dataset, train_frac, val_frac, batch_size):
    tot_len = len(dataset)

    train, val, test = torch.utils.data.random_split(dataset, [int(tot_len*train_frac), int(tot_len*val_frac), tot_len - int(tot_len*train_frac) -  int(tot_len*val_frac)])
    
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader
    
def validate(model, dataloader, prior_mean, prior_var, n_topics, sparse_ten = False):
    val_loss_lis = []
    val_nl_lis = []
    val_kld_lis = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            
            try: 
              sample_encode, sample_decode = batch 
            except:
              sample_encode = sample_decode = batch 
              
            sample_encode = sample_encode.to_dense().float().to(device)
            sample_decode = sample_decode.to_dense().float().to(device)
               
                
            log_recon, posterior_mean, posterior_logvar = model(sample_encode)

            loss , NL, KLD = loss_elbo(input = sample_decode, log_recon = log_recon, posterior_mean = posterior_mean, posterior_logvar = posterior_logvar,
                            prior_mean = prior_mean, prior_var = prior_var, n_topics = n_topics, reg = 1e-10)

            val_loss_lis.append(loss.cpu().detach())
            val_nl_lis.append(NL.cpu().detach())
            val_kld_lis.append(KLD.cpu().detach())


    return np.mean(np.array(val_loss_lis)), np.mean(np.array(val_nl_lis)), np.mean(np.array(val_kld_lis))
    
def validate_median(model, dataloader, prior_mean, prior_var, n_topics, sparse_ten = False):
    val_loss_lis = []
    val_nl_lis = []
    val_kld_lis = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            try: 
              sample_encode, sample_decode = batch 
            except:
              sample_encode = sample_decode = batch
            sample_encode = sample_encode.to_dense().float().to(device)
            sample_decode = sample_decode.to_dense().float().to(device)
               
                
            log_recon, posterior_mean, posterior_logvar = model(sample_encode)

            loss , NL, KLD = loss_elbo(input = sample_decode, log_recon = log_recon, posterior_mean = posterior_mean, posterior_logvar = posterior_logvar,
                            prior_mean = prior_mean, prior_var = prior_var, n_topics = n_topics, reg = 1e-10)

            val_loss_lis.append(loss.cpu().detach())
            val_nl_lis.append(NL.cpu().detach())
            val_kld_lis.append(KLD.cpu().detach())


    return np.median(np.array(val_loss_lis)), np.median(np.array(val_nl_lis)), np.median(np.array(val_kld_lis))


def train_loop(model, optimizer1, optimizer2, trainset, valset, print_mod, device, n_epochs, save_path, config, sparse_ten = True, clip_value = 1e5):
    """
    train the model 
    Args: 
        model: The TLDA model to train
        optimizer1: The optimizer for the encoder
        optimizer2: The optimizer fot hte topic-specific normal distributions
        trainset: The dataset to train on 
        valset: The dataset to use for validation
        print_mod: Number of epochs to print result after 
        device: Either "cpu" or "cuda"
        n_epochs: Number of epochs to train 
        save_path: Path to save the model's state dict
        config: config file from the model to train
        sparse_ten (bool): if a sparse tensor is used for each batch
        clip_value: Above which euclidian norm of the gradient to clip it
    """
    if config.early_stopping == True:
      n_early_stopping = config.n_epochs_early_stopping
      past_val_losses = []

    loss_lis = []
    nl_lis   = []
    kld_lis  = []
    
    loss_lis_all = []
    nl_lis_all   = []
    kld_lis_all  = []
    
    val_loss_lis_all = []
    val_nl_lis_all   = []
    val_kld_lis_all  = []
    
    grad_norm_lis = []
    
    model.train()
    for epoch in range(n_epochs):
      start = time.time()
      for iter, batch in enumerate(trainset):
        
        try:
          sample_encode, sample_decode = batch 
        except:
          sample_encode = sample_decode = batch

        sample_encode = sample_encode.to_dense().float().to(device)
        sample_decode = sample_decode.to_dense().float().to(device)
        

    
        log_recon, posterior_mean, posterior_logvar = model(sample_encode)
    
        loss , NL, KLD = loss_elbo(input = sample_decode, log_recon = log_recon, posterior_mean = posterior_mean, posterior_logvar = posterior_logvar,
                         prior_mean = model.prior_mean, prior_var = model.prior_variance, n_topics = config.n_topics, reg = 1e-10)
        
        optimizer1.zero_grad()       # clear previous gradients
        optimizer2.zero_grad()
        
        loss.backward()             # backprop
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer1.step()
        optimizer2.step()
    
        loss_lis.append(loss.cpu().detach())
        nl_lis.append(NL.cpu().detach())
        kld_lis.append(KLD.cpu().detach())
        
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norm_lis.append(total_norm)
        
        
    
    
      if epoch % print_mod == 0:
    
        end = time.time()
        time_delta = end - start 
    
        mean_loss = np.mean(np.array(loss_lis))
        mean_nl = np.mean(np.array(nl_lis))
        mean_kld = np.mean(np.array(kld_lis))
        
        median_loss = np.median(np.array(loss_lis))
        median_nl = np.median(np.array(nl_lis))

        median_kld = np.median(np.array(kld_lis))
    
        loss_lis_all += loss_lis
        nl_lis_all += nl_lis
        kld_lis_all += kld_lis
    
        loss_lis = []
        nl_lis = []
        kld_lis = []
    
        val_loss, val_nl, val_kld = validate(model, valset, model.prior_mean, model.prior_variance, n_topics = config.n_topics, sparse_ten = True)
        
        val_loss_median, val_nl_median, val_kld_median = validate_median(model, valset, model.prior_mean, model.prior_variance, n_topics = config.n_topics, sparse_ten = True)
        
    
        val_loss_lis_all.append(val_loss)
        val_nl_lis_all.append(val_nl)
        val_kld_lis_all.append(val_kld)
    
    
        print(f'Epoch nr {epoch}: mean_train_loss = {mean_loss}, mean_train_nl = {mean_nl}, mean_train_kld = {mean_kld}, elapsed time: {time_delta}')
        print(f'Epoch nr {epoch}: median_train_loss = {median_loss}, median_train_nl = {median_nl}, median_train_kld = {median_kld}, elapsed time: {time_delta}')
        print(f'Epoch nr {epoch}: mean_val_loss = {val_loss}, mean_val_nl = {val_nl}, mean_val_kld = {val_kld}')
        print(f'Epoch nr {epoch}: median_val_loss = {val_loss_median}, median_val_nl = {val_nl_median}, median_val_kld = {val_kld_median}')
        
        mean_grad_norm = np.mean(np.array(grad_norm_lis))
        max_grad_norm = np.max(np.array(grad_norm_lis))
        median_grad_norm = np.median(np.array(grad_norm_lis))
        grad_norm_lis = []
        print(f'gradient norm: mean: {mean_grad_norm}, median: {median_grad_norm}, max: {max_grad_norm}')
        print()
        print()

        # early stopping based on median validation loss:
        if config.early_stopping:
          if len(past_val_losses) >= n_early_stopping:
            if val_loss_median > max(past_val_losses):
              print(f"Early stopping because the median validation loss has not decreased since the last {n_early_stopping} epochs")
              return loss_lis_all, val_loss_lis_all
            else:
              past_val_losses = past_val_losses[1:] + [val_loss_median]
          else:
            past_val_losses = past_val_losses + [val_loss_median]
        
        
        
    
        torch.save(model.state_dict(), save_path)
        
    return loss_lis_all, val_loss_lis_all
    
    
def smooth_loss(data, window = 100 ):
    """
    smooth the loss
    """
    if isinstance(data, list):
        data = np.array(data)
    
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out
    

def get_topwords(n_topwords, mus_res, L_lower_res, D_log_res, emb_vocab_mat, idx2word, config):
    """
    Compute the topwords according to the paramters of the TLDA model
    
    Args: 
        n_topwords: Number of topwords per topic
        mus_res: means of topics
        L_lower_res: Matrix parametrizing the covariance matrix
        D_log_res: Log of diagonal to parametrize the covariance matrix
        emb_vocab_mat: Matrix with embeddings of each word in the vocabulary, where the words are sorted alphabetically
        idx2word: maps each index to the word
        config: config dict for the model
        
    Return a numpy array of shape (n_topics, n_topwords) that contains the topwords of each topic
    """
    
    probs1 = torch.exp(calc_beta(mus_res, L_lower_res, D_log_res, emb_vocab_mat, config))
    args1 = np.argsort(-probs1.detach().cpu().numpy(), axis = 1)
    
    vocab_arr = np.array(sorted(list(idx2word.values())))
    words1_sort = np.empty(args1.shape, dtype = vocab_arr.dtype)
    
    for t in range(config.n_topics):
      words1_sort[t] = vocab_arr[args1[t]]
    
    probs1_sort = np.empty(probs1.shape)
    
    for i in range(len(probs1)):
      probs1_sort[i] = probs1[i].detach().cpu()[args1[i]]
      
    
    return words1_sort, probs1_sort
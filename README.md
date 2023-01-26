# TNTM

This repository contains the code for the **T**ransformer-Representation **N**eural **T**opic **M**odel (TNTM) based on the paper "Probabilistic Topic Modelling with Transformer
Representations" by Arik Reuter, Anton Thielmann, Christoph Weisser, Benjamin Säfken and Thomas Kneib. 
Our model combines the benefits of topic representations in transformer-based embedding spaces and probabilistic modelling as in, for example, Latent Dirichlet Allocation. 
Therefore, this approach unifies the powerful and versatile notion of topics based on transformer embeddings with probabilistic modelling. 
TNTM is able to find coherent and diverse topics in large datasets and is able to extract large numbers of topics without degrading performance. 



### Preprocessing

The folder Data/ contains an example notebook for obtaining the 20 Newsgroups corpus (https://github.com/MIND-Lab/OCTIS) and the Reuters corpus (https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection).

In Code/Preprocessing, we inlcuded two Jupyter notebooks exemplifying the potential further processing of those two corpora in order to obtain BERT-based word embeddings. 


### Topic modelling with TNTM 

The folder TNTM/ contains the Bag-of-words-based version of TNTM (TNTM_bow.py) and the Sentence-transformer-based version (TNTM_SentenceTransformer.py). 
The syntax of using either closely mirrors the default scikit-learn syntax where the model has to be initialized and subsequently a fit-method can be called.

Here is an example for fitting TNTM with a SentenceTransformer 
```python
tntm = TNTM_SentenceTransformer.TNTM_SentenceTransformer(
      n_topics  = 20, 
      save_path = f"example/{20}_topics", 
      enc_lr    = 1e-3,
      dec_lr    = 1e-3
      )
result = tntm.fit(
              corpus              = corpus,
              vocab               = vocab, 
              word_embeddings     = word_embeddings,
              document_embeddings = document_embeddings)
```
Arguments in this example:
- "n_topics": The number of topics to fit
- "save_path": Where to save the results of the model fitting
- "enc_lr": Learning rate for the encoder of the VAE underlying TNTM
- "dec_lr": Learning rate of the decoder; Used to optimize the paramters of the Gaussians defining the topics

- "corpus": A list of all documents in the corpus where each docuement itself is represented as a list of its words.
- "vocab": List of unique words in the corpus
- "word_embeddings": Embeddings of each vocabulary where the i-th row is the embedding of the i-th word in the vocabulary. Has shape (len(vocab), embedding_dim). Has type torch.Tensor
- "document_embeddings": Embedding of each document in the corpus. Has shape (n_document, embedding_dim) has type torch.tensor

Return values, tuple of: 
- First, a matrix of shape (number_topics, len(vocab)) where each row contains the words of the topic sorted by their likelihood under the topic. I.e. the word at index (5,2) contains the third most likely word for topic 6.
- Second, a matrix of shape (number_topics, len(vocab)) that contains the relative likelihood of each word under the topic of the respective row

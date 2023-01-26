import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
from collections import Counter
import torch 

class Preprocessor:

    def __init__(self, dataset_raw):
        """
        :param list dataset_raw: The unprocessed corpus, given as a list of strings where each list represents a document 
        """
        self.dataset_lis = dataset_raw

    
    def remove_punctuation(self):
        """
        remove all chars that are not letters or numbers  
        """ 
        
        self.dataset_lis = [re.sub(r'[^a-zA-Z\d\s]', u'', text_str, flags=re.UNICODE) for text_str in self.dataset_lis]

    def create_dataframe(self):
        """
        create dataframe that contains a row for each word and the index of the document the word belongs to 
        """

        words_lis = []
        doc_idx_lis = []
        idx = 0
        for text in self.dataset_lis:
            words = text.split()
            doc_idx = [idx for _ in range(len(words))]
            idx += 1
            words_lis += words
            doc_idx_lis +=doc_idx

        df_dict = {'word': words_lis, 'doc_idx': doc_idx_lis}
        df = pd.DataFrame(df_dict)

        self.df = df

    def lowercase(self):
        """
        add column to dataframe that contains words lowercased
        """
        self.df['word_lower'] = self.df['word'].apply(lambda s: s.lower())

    def identify_stopwords(self):
        """
        Add columns that identifies stopwords according to the NLTK package
        """
        nltk.download('stopwords')
        
        sw = set(stopwords.words('english'))
        is_no_stopword  = [False if word in sw else True for word in self.df['word_lower']]
        self.df['is_no_stopword'] = is_no_stopword


    def min_freq_thresh(self, min_freq = 5):
        """
        Remove all words that do not occur at least min_freq times in the corpus
        """
        my_counter = Counter(self.df['word_lower'])
        freq_word = dict(my_counter)
        is_freq = [True if freq_word[word] >= min_freq else False for word in self.df['word_lower']]
        self.df['is_freq'] = is_freq
        
    def max_freq_thresh(self, max_freq_pos = 20):
        """
        Remove the most frequent words within the document 
        """
        my_counter = Counter(self.df['word_lower'])
        freq_word = dict(my_counter)

        most_freq_words = sorted(freq_word, key=freq_word.get, reverse=True)
        too_frq_words = set(most_freq_words[:max_freq_pos])

        is_not_too_freq = [True if word not in too_frq_words else False for word in self.df['word_lower']]
        self.df['is_not_too_freq'] = is_not_too_freq
        

    def remove_short_words(self, min_len = 2):
        """
        remove all words shorter that min_len
        """

        is_not_short = [True if len(word) >= min_len else False for word in self.df['word_lower']]
        self.df['is_not_short'] = is_not_short

    def eval_all(self):
        """
        Add column that is true if every other column is true
        """

        self.df['is_all'] = self.df['is_no_stopword'] & self.df['is_not_short'] & self.df['is_freq'] & self.df['is_not_too_freq']

    def preprocess(self, min_freq = 5, max_freq_pos = 20, min_len = 2):
        """
        preprocess data by calling all the available methods
        """
        print('preprocessing...')
        self.remove_punctuation()
        self.create_dataframe()
        self.lowercase()
        self.identify_stopwords()
        self.min_freq_thresh(min_freq)
        self.max_freq_thresh(max_freq_pos)
        self.remove_short_words(min_len)
        self.eval_all()

        return self.df

    
    def tokenize(self, tokenizer_word):
        """
        Tokenize all the words in the dataframe
        :tokenizer_word: function that takes a word an return the corresponding token(s) in a list (even if it is just one token that is 
        returned)
        
        """
        print('tokenizing...')
        tqdm.pandas()
        word_tokens = self.df['word_lower'].progress_apply(tokenizer_word)

        self.df['word_tokens'] = word_tokens

    def create_doc_lis(self, column, selection_condition):
        """
        create a list that contains the lists of words/tokens belonging to a document according to the 
        'doc_idx' column
        """
        if selection_condition is not None: 
            df_cur = self.df[selection_condition]
            
        else: 
            df_cur = self.df
            
        

        selected_column = df_cur[column]
        doc_idx_column = df_cur['doc_idx']

        max_idx = np.max(doc_idx_column)  #maximum index 
        
        all_doc_lis = []
        for doc_idx in tqdm(range(max_idx + 1)):
            doc_series = selected_column[doc_idx_column == doc_idx]
            
            doc_lis = []
            for elem in doc_series: 
                if isinstance(elem, list):
                    doc_lis += elem
                else: 
                    doc_lis += [elem]

            all_doc_lis.append(doc_lis)

        return all_doc_lis

    def create_chunks(doc_lis_lis, chunk_size):
        """
        take a list of lists of words for each document and partition them into chunks such that 
        transformers can deal with them 
        """
        chunk_lis_lis = []
        for doc_lis in tqdm(doc_lis_lis):
            chunk_lis = []
            for i in range(0, len(doc_lis)+1, chunk_size):
                chunk_lis.append(doc_lis[i: min(i+chunk_size, len(doc_lis))])

            chunk_lis_lis.append(chunk_lis)

        return chunk_lis_lis

    def embed_chunk_lis(chunk_lis_lis, transformer, device):
        """
        Take a list of lists of chunks of tokens and embed the tokens. 
        Return a list of lists of embeddings of the words of each document 
        """
        chunk_emb_ten_lis = []
        for chunk_lis in tqdm(chunk_lis_lis):

            chunk_emb_lis = []
            for chunk in chunk_lis:
  
                emb = transformer(chunk)
                chunk_emb_lis.append(emb)
            chunk_ten = torch.cat(chunk_emb_lis, dim = 0)
            chunk_emb_ten_lis.append(chunk_ten)

        return chunk_emb_ten_lis
        
        
    
    def embeddings_to_word_df(self, embedding_ten_lis):
        """
        add the emebedding of each token back to the dataframe where embedding_ten_lis is a list of tensors
        such that each tensor comprises the embeddings of a chunk of tokens 
        """
        
        emb_lis_lis = []
        
        df_idx = 0
        n_tokens_idx = len(self.df['word_tokens'][df_idx])
        emb_lis = []
        for emb_chunk_ten in tqdm(embedding_ten_lis): 
            for embedding in emb_chunk_ten: 
            
                if n_tokens_idx > 1:
                    emb_lis.append(embedding)
                    
                    n_tokens_idx = n_tokens_idx - 1
                elif df_idx< len(self.df): 
                    emb_lis.append(embedding)
            
                    emb_lis_lis.append(emb_lis)
                    
                    df_idx +=1 
                    
                    if df_idx < len(self.df):
                        n_tokens_idx = len(self.df['word_tokens'][df_idx])
                        emb_lis = []
                    
                    
        self.df['embeddings'] = emb_lis_lis
        return emb_lis_lis
        
        
    def obtain_word_df(self):
        """
        Compute dataframe with embedding for each word, which is obtained by averaging all tokens belonging to a word
        needs self.df with embeddings for each token 
        """
        
        vocab = list(set(self.df['word_lower']))
        
        
        new_df_lis = []
        
        try:
          df_cur = self.df[['embeddings', 'is_all', 'word_lower']]
        except KeyError:
          self.df['is_all'] = True
          df_cur = self.df[['embeddings', 'is_all', 'word_lower']]

        df_cur.set_index('word_lower', inplace = True)
        df_cur.sort_index(inplace =True)
        
        for word in tqdm(vocab):
            
            self_df = df_cur.loc[str(word)]
            
            emb_df = self_df['embeddings']
            
        
            join_emb_lis = []
            for elem in emb_df:
                if isinstance(elem, list):
                    join_emb_lis += elem 
                else:
                    join_emb_lis += [elem]
                    
            join_emb_ten = torch.stack(join_emb_lis)
            
            word_mean = torch.mean(join_emb_ten, dim = 0)
            
            bool_word_df = self_df['is_all']
            
            is_valid = np.all(np.array(bool_word_df))
            
            new_df_lis.append({'word':word, 'embedding':word_mean, "is_valid": is_valid})
            
            
        word_df = pd.DataFrame(new_df_lis)
        word_df = word_df.set_index("word", inplace = False)
        
        self.word_df = word_df
        return word_df
           
            
    def add_mean_embedding_word(self, word_df = None):
        """
        Add the mean embedding contained in word_df of each word to each word.
        If no word_df is passed to this function, just use self.word_df 
        """
        if word_df == None:
            word_df = self.word_df
        
        
        word_series = self.df['word_lower']
        word_series.sort_index()
        
        new_wordemb_lis = []
        for word in tqdm(word_series):
            word_emb = word_df.loc[word]
            new_wordemb_lis.append(word_emb)
            
        self.df['mean_word_embedding'] = new_wordemb_lis
        
    def reindex(self, word_df, cleaned_bow_worddata):
        """
        replace the words by discrete numbers which represent their index in the alphabetically sorted vocabulary
        
        Args: 
            word_df: Dataframe that contains each word
        """
    
        word_df.sort_index(inplace = True)
        word_df['idx'] = list(range(len(word_df)))
        
        
        word2idx = word_df[['idx']].to_dict()['idx']
        
        try: 
            word_df.drop(columns = ['level_0', 'index'])
        except: 
            pass
        word_df = word_df.reset_index(inplace=False)
        idx2word = word_df[['word']].to_dict()
        
        doc_idx_lis_lis = []
        n_bins = len(word2idx)
        for doc in tqdm(cleaned_bow_worddata):
            doc_idx_ten = torch.tensor([word2idx[word] for word in doc], dtype=torch.int64)
            doc_idx_ten_1h = torch.bincount(doc_idx_ten, minlength = n_bins).to_sparse()
            doc_idx_lis_lis.append(doc_idx_ten_1h)
            
        doc_word_1h_ten = torch.stack(doc_idx_lis_lis)
        
        return word2idx, idx2word, doc_word_1h_ten
    
        
        
                        
    def preprocess_and_embed(self, tokenizer_word, transformer_chunk_len, encoder, device, return_values, min_freq = 5, max_freq_pos = 20, min_len = 2):
        """
        Preprocess and embed the entire data 
        
        Args: 
            min_freq (int): Minimal number of documents a word has to be contained in to be considerer valid
            max_freq_pos (int): Maximal rank a word may have according to its frequency among all documents
            min_len (int): Minimal lenth of a word (excluding numbers)
            tokenizer (function): Function that maps a single word onto the LIST tokens representing the word 
            transformer_chunk_len (int): number of tokens the transformer can use at once without sos and eos tokens
            encoder (function): model that maps a chunk of tokens in form of a list onto a tensor of shape (chunk_len, embedding dimension). (This function has to take care of adding sos and eos tokens, padding etc...)
            device (str): 'cuda' or 'cpu'
            return_values (bool): if True, return the computed entities. If False, do not return them as they are already stored as attributes (saves memory)
            
        Returns: 
            The dataframe of the object containing the word, how it was assessed in each cleaning step, its token representation and embedding representation 
            The word-dataframe containing the mean embedding of each word and if the word was assesed as valid during the preprocessing 
            A list comprising the bow-representation of each documt, where the preprocessing has been performed
        """
        
        data_preprocessed = self.preprocess(min_freq=min_freq, max_freq_pos=max_freq_pos, min_len = min_len)   #remove stopwords, punctuation, lowercase, remove too frequent words and to less frequent words and to short words 
        
        print('tokenizing...')
        self.tokenize(tokenizer_word)   #tokenize all the words 
        token_lis = self.create_doc_lis('word_tokens', selection_condition=None)
        
        chunks = Preprocessor.create_chunks(token_lis, transformer_chunk_len)   #split tokens into chunks for transformer 
        
        
        print('compute embeddings of entire corpus...')
        embeddings = Preprocessor.embed_chunk_lis(chunks, encoder, device = device)  #create embeddings 
        
        res2 = self.embeddings_to_word_df(embeddings)   #add embeddings back to dataframe
    
        print('compute mean embeddings of each individual word...')
        self.word_df = self.obtain_word_df()   #create dataframe with embedding for each unique word in the corpus, where the embedding of each workd is the mean of the corresponding tokens
        
        self.add_mean_embedding_word()
        
        print('compute bow representation of each document...')
        self.cleaned_bow_worddata = self.create_doc_lis(column = 'word_lower', selection_condition = self.df['is_all'])  # create list of lists where each list contains the words of a document after cleaning 
        
        
        
        
        return self.df, self.word_df, self.cleaned_bow_worddata
    
    def compute_embeddings(self, tokenizer_word, transformer_chunk_len, encoder, device):
        """
        Only embed the data without further preprocessing
            tokenizer (function): Function that maps a single word onto the LIST tokens representing the word 
            transformer_chunk_len (int): number of tokens the transformer can use at once without sos and eos tokens
            encoder (function): model that maps a chunk of tokens in form of a list onto a tensor of shape (chunk_len, embedding dimension). (This function has to take care of adding sos and eos tokens, padding etc...)
            device (str): 'cuda' or 'cpu'
            return_values (bool): if True, return the computed entities. If False, do not return them as they are already stored as attributes (saves memory)
            
        Returns: 
            The dataframe of the object containing the word, how it was assessed in each cleaning step, its token representation and embedding representation 
            The word-dataframe containing the mean embedding of each word and if the word was assesed as valid during the preprocessing 
            A list comprising the bow-representation of each document, where the preprocessing has been performed
        """
        self.create_dataframe()
        self.lowercase()
        
        print('tokenizing...')
        self.tokenize(tokenizer_word)   #tokenize all the words 
        token_lis = self.create_doc_lis('word_tokens', selection_condition=None)
        
        chunks = Preprocessor.create_chunks(token_lis, transformer_chunk_len)   #split tokens into chunks for transformer 
        
        
        print('compute embeddings of entire corpus...')
        embeddings = Preprocessor.embed_chunk_lis(chunks, encoder, device = device)  #create embeddings 
        
        res2 = self.embeddings_to_word_df(embeddings)   #add embeddings back to dataframe
    
        print('compute mean embeddings of each individual word...')
        self.word_df = self.obtain_word_df()   #create dataframe with embedding for each unique word in the corpus, where the embedding of each workd is the mean of the corresponding tokens
        
        self.add_mean_embedding_word()
        
        print('compute bow representation of each document...')
        self.cleaned_bow_worddata = self.create_doc_lis(column = 'word_lower', selection_condition = self.df['is_all'])  # create list of lists where each list contains the words of a document after cleaning 
        
        self.word_df.sort_values(by = "word", inplace = True)
        
        
        return self.df, self.word_df, self.cleaned_bow_worddata


    def encode_bert_wrap(bert):
        def encode_bert(chunk):
            # take a chunk of tokens and compute their embedding 

            # change chunk
            if len(chunk) < 510:
                
                add_pad = [0 for _ in range(510 - len(chunk))]
                chunk = chunk + add_pad
            else: 
                add_pad = []

            chunk = [101] + chunk + [102]  # add start-of-sequence and end-of-sequence tokens

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            ten_chunk = torch.tensor(chunk).unsqueeze(0).to(device)
            emb = bert(ten_chunk)['last_hidden_state']
            emb = emb[:, 1:-1]

            padding_bool_lis = [True for _ in range(510 - len(add_pad))] + [False for _ in range(len(add_pad))]

            emb = emb[:, padding_bool_lis]

            emb = emb.squeeze(0)
            return emb.detach().cpu()

        return encode_bert
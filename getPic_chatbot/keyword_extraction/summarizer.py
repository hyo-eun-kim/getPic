import numpy as np
from .rank import pagerank
from .word import word_graph
import pandas as pd

class KeywordSummarizer:
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        Tokenize function: tokenize(str) = list of str
    min_count : int
        Minumum frequency of words will be used to construct sentence graph
    window : int
        Word cooccurrence window size. Default is -1.
        '-1' means there is cooccurrence between two words if the words occur in a sentence
    min_cooccurrence : int
        Minimum cooccurrence frequency of two words
    vocab_to_idx : dict or None
        Vocabulary to index mapper
    df : float
        PageRank damping factor
    max_iter : int
        Number of PageRank iterations
    verbose : Boolean
        If True, it shows training progress
    """
    def __init__(self, sents=None, tokenize=None, min_count=2,
        window=-1, min_cooccurrence=2, vocab_to_idx=None,
        df=0.85, max_iter=30, verbose=False):

        self.tokenize = tokenize
        self.min_count = min_count
        self.window = window
        self.min_cooccurrence = min_cooccurrence
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter
        self.verbose = verbose

        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents, bias=None):
        """
        Arguments
        ---------
        sents : list of str
            Sentence list
        bias : None or numpy.ndarray
            PageRank bias term

        Returns
        -------
        None
        """

        g, self.idx_to_vocab = word_graph(sents,
            self.tokenize, self.min_count, self.window,
            self.min_cooccurrence, self.vocab_to_idx, self.verbose) # g는 동시등장빈도 행렬
        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
        if self.verbose:
            print('trained TextRank. n words = {}'.format(self.R.shape[0]))


    def keywords(self, topk=30): #키워드 추출 첫번째 방법
        """
        Arguments
        ---------
        topk : int
            Number of keywords selected from TextRank

        Returns
        -------
        keywords : list of tuple
            Each tuple stands for (word, rank)
        """
        keyword_dict = pd.read_pickle('Data/keyword_dict.pickle')
        if not hasattr(self, 'R'):
            raise RuntimeError('Train textrank first or use summarize function')

        # 크롤링한 데이터셋에서 너무 많이 등장했던 keyword는 불용어로 처리
        for i in range(len(self.R)):
            try:
                if keyword_dict[self.idx_to_vocab[i]] > 500:
                    self.idx_to_vocab = self.idx_to_vocab[:i] + self.idx_to_vocab[i+1:]
                    self.R = np.delete(self.R, i)
            except:
                pass
        idxs = self.R.argsort()[-topk:]
        keywords = [self.idx_to_vocab[idx] for idx in idxs[::-1]]
        return keywords


    def summarize(self, sents, topk=30):
        """
        Arguments
        ---------
        sents : list of str
            Sentence list
        topk : int
            Number of keywords selected from TextRank

        Returns
        -------
        keywords : list of tuple
            Each tuple stands for (word, rank)
        """

        self.train_textrank(sents)
        return self.keywords(topk)


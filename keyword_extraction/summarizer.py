import numpy as np
from .rank import pagerank
from .sentence import sent_graph
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
        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1) #  R은? 단어집합개수 차원의 벡터 #각 숫자가 의미하는것은?
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
        keyword_dict = pd.read_pickle('Data/keyword_dict.pickle') #keyword_dict는 전체 단어집합? #이 딕셔너리의 value값의 의미가 무엇인가
        if not hasattr(self, 'R'):
            raise RuntimeError('Train textrank first or use summarize function')

        # 방법1
        # 크롤링한 데이터셋에서 너무 많이 등장했던 keyword는 불용어로 처리
        # 아직 미완성된 부분으로 임시 필터링 작업
        for i in range(len(self.R)):
            try:
                if keyword_dict[self.idx_to_vocab[i]] > 500: #500의 의미는 무엇인가? #문서에 나온 단어를 하나씩 훑으면서 해당 단어가 keyword_dict에서 500보다 큰 value값을 가질때 
                    self.idx_to_vocab = self.idx_to_vocab[:i] + self.idx_to_vocab[i+1:] #해당단어를 제외하고 idx_to_vocab과
                    self.R = np.delete(self.R, i) #R을 업데이트한다
            except:
                pass
        idxs = self.R.argsort()[-topk:] #argsort: 작은 값부터 순서대로 데이터의 index를 반환해줌(-30:이므로 상위 10개를 뽑는건가?)
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


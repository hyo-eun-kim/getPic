from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class new_KeywordSummarizer:
  def __init__(self, tokenize=None):
    self.tokenize = tokenize

  def new_keyword(self, sents):
    tfidf_vectorizer = TfidfVectorizer(tokenizer = self.tokenize.tokenizer)
    tfidf = tfidf_vectorizer.fit_transform(sents)
    names = tfidf_vectorizer.get_feature_names()
    data = tfidf.todense().tolist()
    # Create a dataframe with the results
    df = pd.DataFrame(data, columns=names)
    result = dict(df.sum(axis=0))
    result = sorted(result.keys(),reverse=True,key=lambda x : result[x])[:10]
    return result

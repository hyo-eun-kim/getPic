from konlpy.tag import Komoran # 웹은 코모란
#from khaiii import KhaiiiApi  챗봇은 khaiii


class Komoran_Tokenizer:
    def __init__(self):
        self.komoran = Komoran() #품사를 잘 구분해주는 코모란 #khaiii와 비슷

    def tokenizer(self, sent):
        sent = sent.replace('\n', ' ')
        sent = sent.replace('"', '')
        words = self.komoran.pos(sent)
        words = [w for w, p in words if p.startswith("N")]  # 명사 token만 추출한다.
        words = [word for word in words if len(word) > 1]
        stopwords = list(set(['아무','이곳', '거기', '이분', '남자', '여자', '저곳','때문','녀석','대답','방법','모습',
                            '감정','곳곳','처음','그녀','자기','가지','본인','만큼','정말',
                            '그때','지금','이름','누구','이때','전날','순간','예전','마찬가지','오늘','내일','요즘','우리',
                            '과정','사람','인생','생각','최대한','개월','노릇','그것','저것','이것','요일','결국','이후','이전','다섯','여섯',
                            '일곱','여덟','아홉','하나','무엇','동안','정도','기간']))
        words = [word for word in words if not word in stopwords]
        return words

# class Khaiii_Tokenizer:
#     def __init__(self):
#         self.khaiii = KhaiiiApi()
#
#     def tokenizer(self, sent):
#         sent = sent.replace('\n', ' ');
#         sent = sent.strip()
#         words = []
#         for word in self.khaiii.analyze(sent):
#             words.extend([morph.lex for morph in word.morphs if morph.tag.startswith('N')] )
#         words = [word for word in words if len(word) > 1]
#         stopwords = list(set(['아무','이곳','거기', '이분', '남자', '여자','저곳','때문','녀석','대답','방법','모습',
#                             '감정','곳곳','처음','그녀','자기','가지','본인','만큼','정말',
#                             '그때','지금','이름','누구','이때','전날','순간','예전','마찬가지','오늘','내일','요즘','우리',
#                             '과정','사람','인생','생각','최대한','개월','노릇','그것','저것','이것','요일','결국','이후','이전','다섯','여섯',
#                             '일곱','여덟','아홉','하나','무엇','동안','정도','기간']))
#         words = [word for word in words if not word in stopwords]
#         return words
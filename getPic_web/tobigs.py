# -*- coding: utf-8 -*-
from keyword_extraction import KeywordSummarizer, new_KeywordSummarizer
from tokenizer.tokenizer import Komoran_Tokenizer
from image_crawling.crawling import Crawling
from sentiment_analysis.bertdata import BERTDataset
from sentiment_analysis.bertclassifier import BERTClassifier
from style_transfer.styletransfer import TransformerNet
from style_transfer.util import *
import kss
import torch
import pandas as pd
import numpy as np
import numexpr as ne
import gluonnlp as nlp
import torchvision.transforms as transforms
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import datetime
import os
import re
import random
# import telegram
import requests
import queue
import urllib
import json

# hyperparameter
max_len = 64
batch_size = 128
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

def get_keyword(input_text):
    try:
        print("----- 키워드 추출중 -----")
        komoran_tokenizer = Komoran_Tokenizer()
        # khaiii_tokenizer = Khaiii_Tokenizer()
        keyword_extractor = KeywordSummarizer(tokenize=komoran_tokenizer)
        keyword = keyword_extractor.summarize(input_text, topk=10)
        if (len(keyword) < 5):
            raise Exception
        return keyword
    except:
        print("여기1")
        try:
            keyword_extractor = new_KeywordSummarizer(tokenize=komoran_tokenizer)
            keyword = keyword_extractor.new_keyword(input_text)
            return keyword
        except:
            print("여기2")
            pass


def get_sentimentLabel(input_text, time_info):
    device = 'cpu'
    try:
        print("----- 모델 불러오기-----")
        bertmodel, vocab = get_pytorch_kobert_model()
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

        weights = torch.load('weight/bert_weight.pth', map_location=torch.device('cpu'))
        model.load_state_dict(weights)

        model = model.to(device)
        model.eval()

        print("----- 감성라벨 예측 -----")
        essay = pd.DataFrame(input_text)
        essay['label'] = 1
        # print("텍스트 df") #임시
        save_link = "Data/{}.txt".format(time_info)
        essay.to_csv(save_link, sep='\t', index_label='idx')
        # dataset_sentences = nlp.data.TSVDataset(save_link)
        dataset_sentences = nlp.data.TSVDataset(save_link, field_indices=[1, 2], num_discard_samples=1)

        print(dataset_sentences)
        print("BERTDataset error?")
        data_sentences = BERTDataset(dataset_sentences, 0, 1, tok, 100, True, False)  # max_len (100)
        print("torch error?")
        sentences_dataloader = torch.utils.data.DataLoader(data_sentences, batch_size=len(data_sentences),
                                                           num_workers=5)
        print("torch 들어가기 직전임")
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(sentences_dataloader):
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                label = label.long().to(device)
                valid_length = valid_length
                outputs = model(token_ids, valid_length, segment_ids)
        pred_test = outputs
        arr = np.array(pred_test.tolist())
        arr = ne.evaluate("exp(arr)")
        print("torch 들어가기 직후임")

        label_dic = dict(
            [(0, 'anger'), (1, 'fear'), (2, 'happy'), (3, 'miss'), (4, 'sad'), (5, 'surprise'), (6, 'worry')])
        for i in range(7):
            essay[label_dic[i]] = [proba[i] for proba in arr]
        essay['label'] = list(map(np.argmax, arr))
        indices = np.array(list(map(np.max, arr))).argsort()[::-1][0:min(len(essay), 10)]
        prob = essay.iloc[indices].sum(axis=0)[2:].astype(float)
        prob['happy'] *= 0.6
        prob['fear'] *= 0.8
        prob['worry'] *= 2
        result = prob.idxmax()
        print("예측라벨 > ", result)
        return result
    except:
        pass


    # print("----- 모델 불러오기 -----")
    # # bertmodel, vocab = get_pytorch_kobert_model(cachedir=r"C:\Users\kimminjeong\kobert\kobert_news_wiki_ko_cased-1087f8699e.spiece")
    # bertmodel, vocab = get_pytorch_kobert_model()
    # tokenizer = get_tokenizer()
    # # tok = nlp.data.BERTSPTokenizer( tokenizer, vocab, lower=False, path = r'C:\Users\kimminjeong\kobert\kobert_news_wiki_ko_cased-1087f8699e.spiece')
    # tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    # device = 'cpu'
    # model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    # # weights = torch.load('weight/bert_weight.pth', map_location='cuda:0')
    # weights = torch.load('weight/bert_weight.pth', map_location=torch.device('cpu'))
    # model.load_state_dict(weights)
    # model = model.to(device)
    # model.eval()
    #
    # print("----- 감성라벨 예측 -----")
    # text = pd.DataFrame(input_text)
    # text['label'] = 1
    # save_link = "Data/{}.txt".format(time_info)
    # text.to_csv(save_link, sep='\t', index_label='idx')
    # dataset_sentences = nlp.data.TSVDataset(save_link, field_indices=[1, 2], num_discard_samples=1)
    # data_sentences = BERTDataset(dataset_sentences, 0, 1, tok, max_len, True, False)  # max_len
    # sentences_dataloader = torch.utils.data.DataLoader(data_sentences, batch_size=len(data_sentences), num_workers=5)
    #
    # with torch.no_grad():
    #     for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(sentences_dataloader):
    #         token_ids = token_ids.long().to(device)
    #         segment_ids = segment_ids.long().to(device)
    #         label = label.long().to(device)
    #         valid_length = valid_length
    #         outputs = model(token_ids, valid_length, segment_ids)
    #
    # pred_label = outputs
    #
    # arr = np.array(pred_label.tolist())
    # arr = ne.evaluate("exp(arr)")
    # # print("확률값은!?? ", arr)
    # text['label'] = list(map(np.argmax, arr))
    # indices = np.array(list(map(np.max, arr))).argsort()[::-1][0:min(len(text), 10)]
    # # print(indices) #상위 10개 문장 인덱스 출력
    # result = text.iloc[indices]['label'].value_counts().keys()[0]
    # label_dic = dict([(0, 'anger'), (1, 'fear'), (2, 'happy'), (3, 'miss'), (4, 'sad'), (5, 'surprise'), (6, 'worry')])
    #
    # print("예측라벨 > ", label_dic[result])
    # return label_dic[result]






def get_crawlingImage(kor_keyword, client_id, client_secret, time_info, now):
    crawling = Crawling(kor_keyword=kor_keyword, client_id=client_id, client_secret=client_secret)
    print("----- 키워드 번역중 -----")
    crawling.get_eng_keyword()
    print("번역된 키워드 > ", crawling.eng_keyword)
    print("----- 크롤링 수행중 -----")
    crawling.crawling_image()
    #crawling.links = ['https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1517971071642-34a2d3ecc9cd?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1488190211105-8b0e65b80b4e?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1472289065668-ce650ac443d2?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1505682634904-d7c8d95cdc50?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1455390582262-044cdead277a?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1555601568-c9e6f328489b?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1508780709619-79562169bc64?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/reserve/LJIZlzHgQ7WPSh5KVTCB_Typewriter.jpg?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1457369804613-52c61a468e7d?ixlib=rb-1.2.1&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1535182481664-c9c02090dc4f?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1498677231914-50deb6ba4217?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1470049384172-927891aad5e9?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1485458029194-00cff7de3ef7?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1523289333742-be1143f6b766?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1455390582262-044cdead277a?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80', 'https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;w=1000&amp;q=80']
    #crawling.caps = ['person writing on brown wooden table near white ceramic mug', 'MacBook Pro near white open book', '_2VWD4 _2zEKz" da', 'person holding ballpoint pen writing on notebook', 'two gray pencils on yellow surface', 'black Corona typewriter on brown wood planks', 'fountain pen on black lined paper', '_2VWD4 _2zEKz" da', 'person using laptop', 'black Fayorit typewriter with printer paper', 'open book lot', 'person writing on brown wooden table near white ceramic mug', 'MacBook Pro near white open book', 'person typing on red typewriter', 'woman in blue chambray long-sleeved top sitting on black leather chair with silver MacBook on lap', 'black pencil on white card beside brown knit textile', 'person in blue dress shirt holding red book and colored pens', 'person holding pencil and stick note beside table', 'fountain pen on black lined paper', 'person writing on brown wooden table near white ceramic mug', 'MacBook Pro near white open book', 'person writing on brown wooden table near white ceramic mug', 'MacBook Pro near white open book', 'person writing on brown wooden table near white ceramic mug', 'MacBook Pro near white open book', 'person writing on brown wooden table near white ceramic mug', 'MacBook Pro near white open book', 'person writing on brown wooden table near white ceramic mug', 'MacBook Pro near white open book', 'person writing on brown wooden table near white ceramic mug', 'MacBook Pro near white open book', 'person writing on brown wooden table near white ceramic mug', 'MacBook Pro near white open book', 'person writing on brown wooden table near white ceramic mug', 'MacBook Pro near white open book']
    print("----- 이미지 저장중 -----")
    link = crawling.save_image(time_info, now)
    return link


def get_finalImage(image_link, sentiment_label, filename):
    device = 'cpu'
    try:
        print("----- GetPic! -----")
        content_image = load_image(image_link, scale=4)  # edit
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        with torch.no_grad():
            weight_list = os.listdir('weight/style_weight/{}'.format(sentiment_label))
            style_weight_path = random.choice(weight_list)
            # path_name = style_weight_path.replace(".pth", "")
            # print("스타일필터 > ", path_name)
            checkpoint = torch.load("weight/style_weight/{}/{}".format(sentiment_label, style_weight_path), map_location=device)

            style_model = TransformerNet()
            style_model.load_state_dict(checkpoint['model_state_dict'])
            style_model.to(device)
            output = style_model(content_image).cpu()
        save_image("static/img/{}.jpg".format(filename), output[0], disp=True)
        print("----- FINISH -----")
    except:
        pass

# def get_finalImage(image_link, sentiment_label, filename):
#     print("----- GetPic! -----")
#     device = 'cpu'
#     content_image = load_image(image_link, scale=2)  # edit
#     content_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.mul(255))
#     ])
#     content_image = content_transform(content_image)
#     content_image = content_image.unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         weight_list = os.listdir('weight/style_weight/{}'.format(sentiment_label))
#         style_weight_path = random.choice(weight_list)
#         print("스타일필터 > ", style_weight_path)
#         checkpoint = torch.load("weight/style_weight/{}/{}".format(sentiment_label, style_weight_path),
#                                 map_location=device)
#
#         style_model = TransformerNet()
#         style_model.load_state_dict(checkpoint['model_state_dict'])
#         style_model.to(device)
#         output = style_model(content_image).cpu()
#     save_image("static/img/{}.jpg".format(filename), output[0], disp=True)
#     print("----- FINISH -----")


if __name__ == '__main__':

    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    # 1. get input text from user
    input_text = input("사연을 입력해주세요 > ")
    time_info = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    input_text = kss.split_sentences(input_text)

    # 2. extract keyword
    keyword = get_keyword(input_text)
    print("추출된 키워드 > ", keyword)
    # keyword = ['글', '영상', '내', '그림', '사진', '수단', '일상', '기록', '글쓰기', '동영상']

    # 3. image crawling
    if len(keyword) < 3:
        pass  # 예외처리
    else:
        image_link = get_crawlingImage(keyword, "KRTSOhiLDjFo8VpVkekS", "PnJAftBpaI", time_info)
        # 4. predict sentiment label
        sentiment_label = get_sentimentLabel(input_text, time_info)
        # 5. style transfer
        get_finalImage(image_link, sentiment_label, time_info)
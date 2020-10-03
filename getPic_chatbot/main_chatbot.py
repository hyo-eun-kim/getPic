# -*- coding: utf-8 -*-
from keyword_extraction import KeywordSummarizer, new_KeywordSummarizer
from tokenizer.tokenizer import Komoran_Tokenizer, Khaiii_Tokenizer
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
import sys
import random
import telegram
import requests
import urllib
import json
import queue


class Keyword_Error(Exception):
    pass

class Sentiment_Error(Exception):
    pass

class Crawling_Error(Exception):
    pass

class Transfer_Error(Exception):
    pass


# 키워드 추출하는 코드 
# input_text : list of sentence
def get_keyword(input_text):
    try:
        print("1. extract keyword from message")
        khaiii_tokenizer = Khaiii_Tokenizer()
        keyword_extractor = KeywordSummarizer(tokenize=khaiii_tokenizer)
        keyword = keyword_extractor.summarize(input_text, topk=10)
        if len(keyword) < 4:
            raise Exception() # change to TF-IDF
        return keyword
    except:
        print("\t change to TF-IDF keyword extraction")
        try:
            keyword_extractor = new_KeywordSummarizer(tokenize=khaiii_tokenizer)
            keyword = keyword_extractor.new_keyword(input_text)
            return keyword
        except:
            raise Keyword_Error()
        

def get_sentimentLabel(input_text, time_info):
    try:
        print("2. predict sentiment label")
        device = torch.device("cpu")
        bertmodel, vocab = get_pytorch_kobert_model()
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

        model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
        weights = torch.load('weight/bert_weight.pth', map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        model = model.to(device)
        model.eval()

        essay = pd.DataFrame(input_text)
        essay['label'] = 1
        save_link = "Data/{}.txt".format(time_info)
        essay.to_csv(save_link, sep='\t', index_label='idx')
        dataset_sentences = nlp.data.TSVDataset(save_link, field_indices=[1, 2], num_discard_samples=1)
        data_sentences = BERTDataset(dataset_sentences, 0, 1, tok, 100, True, False) # max_len (100)
        sentences_dataloader = torch.utils.data.DataLoader(data_sentences, batch_size=len(data_sentences), num_workers=5)

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

        label_dic = dict([(0, 'anger'), (1, 'fear'), (2, 'happiness'), (3, 'miss'), (4, 'sadness'), (5, 'surprised'), (6, 'worry')])
        for i in range(7):
            essay[label_dic[i]] = [proba[i] for proba in arr]
        essay['label'] = list(map(np.argmax, arr))
        indices = np.array(list(map(np.max, arr))).argsort()[::-1][0:min(len(essay),10)]
        prob = essay.iloc[indices].sum(axis=0)[2:].astype(float)
        prob['happiness'] *= 0.6
        prob['fear'] *= 0.8
        prob['worry'] *= 2
        result = prob.idxmax()
        if result == 'fear':
            result = 'sadness'
        return result
    except:
        raise Sentiment_Error()


def get_crawlingImage(kor_keyword, client_id, client_secret, time_info):
    try:
        print("3. crawling images")
        crawling = Crawling(kor_keyword=kor_keyword, client_id=client_id, client_secret=client_secret)
        crawling.get_eng_keyword()  # 키워드 번역 수행
        crawling.crawling_image()   # 이미지 크롤링
        candidate_link = crawling.get_candidate_link()
        return candidate_link 
    except:
        raise Crawling_Error()


def get_finalImage(image_link, sentiment_label, time_info):
    try:
        print("4. style transfer")
        device = torch.device("cpu")
        content_image = load_image(image_link, scale=4) # edit
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        with torch.no_grad():
            weight_list = os.listdir('weight/style_weight/{}'.format(sentiment_label))
            style_weight_path = random.choice(weight_list)
            path_name = style_weight_path.replace(".pth", "")
            checkpoint = torch.load("weight/style_weight/{}/{}".format(sentiment_label, style_weight_path), map_location=device)
            
            style_model = TransformerNet()
            style_model.load_state_dict(checkpoint['model_state_dict'])
            style_model.to(device)
            output = style_model(content_image).cpu()
        save_image("result/{}.jpg".format(time_info), output[0], disp=True)
        print("---------- FINISH ----------")
    except:
        content_image = load_image(image_link)
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        with torch.no_grad():
            weight_list = os.listdir('weight/style_weight/{}'.format(sentiment_label))
            style_weight_path = random.choice(weight_list)
            path_name = style_weight_path.replace(".pth", "")
            checkpoint = torch.load("weight/style_weight/{}/{}".format(sentiment_label, style_weight_path), map_location=device)
            
            style_model = TransformerNet()
            style_model.load_state_dict(checkpoint['model_state_dict'])
            style_model.to(device)
            output = style_model(content_image).cpu()
        save_image("result/{}.jpg".format(time_info), output[0], disp=True)


def get_chatbot_content(API_key_Telegram, index) :
    endpoint = 'https://api.telegram.org/bot'
    query = '/' + 'getUpdates'
    URL = endpoint + API_key_Telegram + query

    request = urllib.request.Request(URL)
    response = urllib.request.urlopen(request)
    rescode = response.getcode() 
    request_body = urllib.request.urlopen(request).read()

    if rescode == 200:
        request_json = json.loads(request_body)
        return (
                request_json['result'][index]['message']['from']['id'], 
                request_json['result'][index]['message']['text']) 
    else:
        return (None, None)


def get_chatbot_index(API_key_Telegram):
    endpoint = 'https://api.telegram.org/bot'
    query = '/' + 'getUpdates'
    URL = endpoint + API_key_Telegram + query

    request = urllib.request.Request(URL)
    response = urllib.request.urlopen(request)
    rescode = response.getcode() # 정상이면 200
    request_body = urllib.request.urlopen(request).read()
    
    if rescode == 200:
        request_json = json.loads(request_body)
        return len(request_json['result'])-1
    else:
        return None


def send_message(API_key_Telegram, chat_id, input_text) :
    endpoint = 'https://api.telegram.org/bot'
    action = '/' + 'sendmessage'
    param_list = [
        'chat_id=' + str(chat_id), 
        'text=' + input_text
    ]
    param = '&'.join(param_list)
    query = action + '?' + param
    URL = endpoint + API_key_Telegram + query
    try : 
        r = requests.get(URL)
    except Exception as e:
        print(str(e))



def main(argv):
    if len(argv) != 4:
        print('"main.py "API_key_Telegram" "NAVER PAPAGO client id" "NAVER PAPAGO, NAVER PAPAGO clienct secret"')
        return 1;

    _, API_key_Telegram, client_id, client_secret = argv
    bot = telegram.Bot(token=API_key_Telegram)

    GetPic_queue = queue.Queue()
    latest_index = get_chatbot_index(API_key_Telegram)
    print("ＧＥＴＰＩＣ　ＳＴＡＲＴ")
    while True:
        while (latest_index != get_chatbot_index(API_key_Telegram)):
            latest_index += 1
            user_id, user_message = get_chatbot_content(API_key_Telegram, latest_index)
            GetPic_queue.put([user_id, user_message])
            print("0. new message! ")

        while (GetPic_queue.qsize() != 0):
            try:
                # 1. get message from queue
                user_id, user_message = GetPic_queue.get()
                user_message = kss.split_sentences(user_message)
                user_message = [message.replace('\n', ' ') for message in user_message]
                time_info = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

                # 2. extract keyword
                send_message(API_key_Telegram, user_id, "오호랏~! 멋진 글이네요! 제가 글을 읽어볼게요!")
                keyword = get_keyword(user_message)  # 문장으로 이루어진 리스트
                send_message(API_key_Telegram, user_id, "이 글은 " + ", ".join(keyword[:5]) + "에 관한 글이네요?")
                print("keyword > ", keyword)

                # 3. predict sentiment label
                send_message(API_key_Telegram, user_id, "기다리는 동안 지루하지 않게 재밌는 퀴즈를 준비해보았어요ㅎㅎ 맞혀보시겠어요?")
                send_message(API_key_Telegram, user_id, "여름마다 일어나는 전쟁은 무엇일까요~?")
                sentiment_label = get_sentimentLabel(user_message, time_info)
                send_message(API_key_Telegram, user_id, "더워(the war)")
                send_message(API_key_Telegram, user_id, "깔깔깔~ 너무 재밌죠~~~? 앗, 너무 어렵다구요?!;; \n그렇다면 시간을 좀 더 넉넉하게 드려볼게요~!")

                # 4. image crawling
                send_message(API_key_Telegram, user_id, "아마존에 살고 있는 사람의 이름은 무엇일까요?")
                candidate_link= get_crawlingImage(keyword, client_id, client_secret, time_info)
                download_link = random.choice(candidate_link)
                save_link = "Data/{}.jpg".format(time_info)
                urllib.request.urlretrieve(download_link, save_link)  # 다운받는 주소, 저장할 주소
                send_message(API_key_Telegram, user_id, "아마...존?")

                # 5. style transfer
                send_message(API_key_Telegram, user_id, "마지막 문제! 돌잔치를 영어로 하면 무엇일까요?")
                get_finalImage(save_link, sentiment_label, time_info)
                send_message(API_key_Telegram, user_id, "락페스티벌(돌-rock 잔치-festival)! 푸하하~!")

                # 6. send image to user
                bot.send_photo(user_id, open("result/{}.jpg".format(time_info), 'rb'))
                send_message(API_key_Telegram, user_id, "우와~ 드디어 완성되었어요~!")

            except Keyword_Error:
                print(Keyword_Error)
                send_message(API_key_Telegram, user_id, "글이 짧아 키워드를 추출할 수 없네요ㅜㅜ 다른 글은 없나요?")
                continue
            except Crawling_Error:
                print(Crawling_Error)
                send_message(API_key_Telegram, user_id, "글과 어울리는 그림을 찾지 못했어요.. 혹시 다른 글은 없나요?")
                continue
            except Sentiment_Error:
                print(Sentiment_Error)
                send_message(API_key_Telegram, user_id, "글의 감성을 파악하는 과정에서 문제가 발생했어요!! 혹시 다른 글은 없나요?")
                continue
            except Transfer_Error:
                print(Transfer_Error)
                send_message(API_key_Telegram, user_id, "이미지를 변환하는 과정에서 문제가 발생했어요! 다시 한 번 시도해보시겠어요?")
                continue
            except:
                print("문제 발생")


if __name__ == '__main__':
    main(sys.argv)


            


    


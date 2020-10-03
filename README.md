# GetPic
**Tobigs 10th Conference**

(배너 png파일 주소 들어갈자리)

## Table of Contents
- [1. Short Description](#1-short-description)
  - [Example](#example)
- [2. ChatBot](#2-chatbot)
  - [2.1. Install](#21-install)
  - [2.2. Test](#22-test)
- [3. Web](#3-web)
  - [3.1. Getting Started in Web](#31-getting-started-in-web)
  - [3.2. Install](#32-install)
  - [3.3. Test](#33-test)
- [4. Contributing](#4-contributing)
  

---

## 1. Short Description

**getPic** is a service that create illustration based on the input writing.
<br/>
The illustration is made based on the writing's context and emotion.
<br/>
getPic is a project developed by ToBigs Team3 using [KoBERT](https://github.com/SKTBrain/KoBERT/blob/master/README.md), Komoran, [Kwaii](https://github.com/kakao/khaiii.git), [TextRank](https://github.com/lovit/textrank/tree/master/textrank), [Fast-style Transfer](https://hoya012.github.io/blog/Fast-Style-Transfer-Tutorial/), etc.
<br/>
**getPic** is both available in **Web** and **Chatbot**.
So feel free to try!

### Example (ChatBot)


### Example (Web)

<table>
  <tr>
    <td align="center"><img src="https://user-images.githubusercontent.com/68496320/94986606-08c30380-059b-11eb-8b4b-6800f304aba2.png" width="500px;" alt=""/></a></td>
      <td align="center"><img src="https://user-images.githubusercontent.com/68496320/94986607-09f43080-059b-11eb-97af-aefc53af45bd.png" width="500px;" alt=""/></a></td>
  </tr>
</table>

---

  
## 2. ChatBot

### 2.0. Install

Get access to [the drive](https://drive.google.com/drive/u/1/folders/1qkN8eAyB-1318YG-4d-BpyslDmhI23dI), download 'bert_weight.pth' file, and put the file into the location `ChatBot/weight/`

[이 드라이브](https://drive.google.com/drive/u/1/folders/1qkN8eAyB-1318YG-4d-BpyslDmhI23dI)에 접속하여 `bert_weight.pth` 파일을 다운로드 받아 `ChatBot/weight/` 경로에 넣어주시기 바랍니다.

### 2.1. Colab
- We recommend using GPU in Colab. You can change the runtime type by :[Runtime]-[Change runtime type]-[GPU] 
<br/>
- Colab에서 [런타임] - [런타임 유형 변경] - 하드웨어 가속기(GPU) 사용을 권장합니다.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)

---

## 3. Web

### 3.1 At the Anaconda Prompt
- Create virtual environment
```sh
conda create -n virtual_environment_name
```
- Activate virtual environment
```sh
conda activate virtual_environment_name
```
- Install requirements
```sh
pip install -r requirements.txt
```
### 3.2 At the PyCharm
- Activate virtual environment
```python
conda activate virtual_environment_name
```
- Execution
```python
python main_web.py
```
---

## 4. Contributing

<table>
  <tr>
    <td align="center"><a href="https://github.com/yunkio"><img src="https://user-images.githubusercontent.com/48192546/94985703-c4803500-0593-11eb-8912-341bf38e9fa4.jpg" width="100px;" alt=""/><br /><sub><b>Kio Yun</b></sub> </a></td>
      <td align="center"><a href="https://github.com/hyo-eun-kim"><img src="https://user-images.githubusercontent.com/48192546/94985814-a961f500-0594-11eb-81e0-ca1b0985adcf.jpg" width="100px;" alt=""/><br /><sub><b>Hyoeun Kim</b></sub> </a></td>
      <td align="center"><a href="https://github.com/KimHyunsun"><img src="https://user-images.githubusercontent.com/48192546/94985822-b0890300-0594-11eb-93a6-0d7965ea55dc.jpg" width="100px;" alt=""/><br /><sub><b>Hyunsun Kim</b></sub> </a></td>
    <td align="center"><a href="https://github.com/kmmnjng528"><img src="https://user-images.githubusercontent.com/48192546/94985889-28572d80-0595-11eb-98df-301edd4b764d.jpg" width="100px;" alt=""/><br /><sub><b>Minjeong Kim</b></sub> </a></td>
    <td align="center"><a href="https://github.com/gyeong707"><img src="https://user-images.githubusercontent.com/48192546/94985892-2db47800-0595-11eb-80dd-1f053acc0fc9.jpg" width="100px;" alt=""/><br /><sub><b>Migyeong Kang</b></sub> </a></td>
    <td align="center"><a href="https://github.com/602-go"><img src="https://user-images.githubusercontent.com/48192546/94985894-3016d200-0595-11eb-9159-5db2eb4de0c5.jpg" width="100px;" alt=""/><br /><sub><b>Yookyung Kho</b></sub> </a></td>
  </tr>
</table>

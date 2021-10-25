# Korean-Story-Generator
---
> #### 사전학습된 GPT2를 동화 데이터로 파인튜닝해 동화 생성하는 모델..
>> colab test용 .ipynb 파일 2개.
>> 
>> main.py실행시 train 또는 텍스트 생성
>>> - cfg/config_json.json 의 "train_flg": true 값에 따라 바뀜.
>>> 
>>> - true면 train, false면 텍스트 생성 수행.
>
> ```pip install -r requirements.txt``` 이용 환경 셋팅
>
> 사전학습 모델은 SKT의 KoGPT2 Ver 2.0 사용
> - [SKT](https://github.com/SKT-AI/KoGPT2)
> 
> 파인튜닝과 데이터셋 로더등 기본적인 구조NarrativeKoGPT2를 참조 하여 사용
> - [NarrativeKoGPT2](https://github.com/shbictai/narrativeKoGPT2)
>
> 뽀로로등 등장인물의 이름을 토크나이저가 쪼개는 문제 발생
> - 토크나이저와 사전의 unused토큰에 뽀로로 등장인물 추가하여 해결함.
>
> 토크나이저 경로 수정
> - 기존은 허깅페이스에 있는 skt koGPT2 v2의 사전학습모델의 사전을 사용했지만 뽀로로 등장인물을 추가해 수정한것으로 변경
> 
> 정상적인 동작 수행 확인.

# Korean-Story-Generator
> 사전학습된 GPT2를 동화 데이터로 파인튜닝해 동화 생성하는 모델..
>> colab test용 .ipynb 파일 2개.
>> 
>> main.py실행시 train 또는 텍스트 생성
> - cfg/config_json.json 의 "train_flg": true 값에 따라 바뀜.
> 
> - true면 train, false면 텍스트 생성 수행.
>
> ```pip install -r requirements.txt``` 이용 환경 셋팅
>
> 사전학습 모델은 SKT의 KoGPT2 Ver 2.0 사용
> - [SKT](https://github.com/SKT-AI/KoGPT2)
> 
> 파인튜닝과 데이터셋 로더등 기본적인 구조NarrativeKoGPT2를 참조 하여 사용
> - [NarrativeKoGPT2](https://github.com/shbictai/narrativeKoGPT2)

# Korean-Story-Generator
---
- 훈련된 모델은 [링크](https://drive.google.com/drive/folders/1xPkj4Xd5DrvpeoA0xjXK1k7Fm57E6zZF?usp=sharing)의 Korean-Story-Generator압축파일을 내려받아 사용할 수 있다.
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
>
---
> ### 추후 사용을 위해선.
>> data 폴더에 데이터셋을 구성해야한다( .txt파일)
>> - 동화별 구분은 줄바꿈을 이용하여 구분하고 제목과 내용은 (제목 : 내용)과 같은 형식으로 구분하여 저장한다.
>> ```
>> 나무꾼 : 나무꾼이 나무를 한다 ....
>> 늑대와 돼지 : 늑대와 돼지가 있었어요 .....
>> ```
>> 토크나이저와 사전의 unused 토큰 추가 혹은 변경은 [링크](https://github.com/minchan5224/TIL/blob/main/Multicampus/Project/%EC%B5%9C%EC%A2%85%ED%94%8C%EC%A0%9D/1024_%EB%8B%A4%EC%8B%9C_%ED%8C%8C%EC%9D%B8%ED%8A%9C%EB%8B%9D.md)를 참조한다.
>> 
>>

import random
import json
import torch
from torch.utils.data import DataLoader # 데이터로더
from gluonnlp.data import SentencepieceTokenizer 
from model.torch_gpt2 import GPT2Config, GPT2LMHeadModel # model폴더의 torch_gpt2.py의 
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from util.data import FairyDataset
import gluonnlp
from tqdm import tqdm
import torch, gc

def model_load(model_path, kogpt2_config):
    if model_path == '':
        # KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
        kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
        # model_path로부터 다운로드 받은 내용을 load_state_dict으로 업로드
        kogpt2model.from_pretrained("skt/kogpt2-base-v2")
        # 추가로 학습하기 위해 .train() 사용
        kogpt2model.train()
    else :
        from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
        # Device 설정
        device = torch.device(ctx)
        # 저장한 Checkpoint 불러오기
        checkpoint = torch.load(model_path, map_location=device)
        # KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
        kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
        kogpt2model.load_state_dict(checkpoint['model_state_dict'])
        kogpt2model.eval() # 예측

    vocab_b_obj = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='<s>', eos_token='</s>', unk_token='<unk>',  pad_token='<pad>', mask_token='<mask>')

    return kogpt2model, vocab_b_obj
    
def main(cfg):
    train_flg = cfg['train_flg']
    data_file_path = cfg['data_file_path']
    batch_size = cfg['batch_size']
    epochs=cfg['epochs'] 
    save_model_path = cfg['save_model_path']
    kogpt2_config = cfg['kogpt2_config']

    model, vocab_b_obj = model_load(save_model_path+"gt_checkpoint_2.tar", kogpt2_config)
    vocab, sentencepieceTokenizer = vocab_b_obj.get_vocab(), vocab_b_obj.tokenize

    if train_flg: # GTP2 파인튜닝
        model.train()

        ### 학습 데이터 로드
        if data_file_path == None:
            data_file_path = '/content/drive/MyDrive/textG//data/dataset.txt'
        dataset = FairyDataset(data_file_path, vocab, sentencepieceTokenizer)
        fairy_data_loader = DataLoader(dataset, batch_size=batch_size)

        ### 파라미터 설정
        learning_rates = [1e-4, 5e-5, 2.5e-5, 2e-5]# 학습률 잠시 수정 원래는 1e-5임
        criterion = torch.nn.CrossEntropyLoss()
        

        ### 메모리 초과뜨면 한번씩 지워주기 위해서 사용
        # gc.collect()
        # torch.cuda.empty_cache()


        ### 학습시작
        model.cuda()
        print('KoGPT-2 Transfer Learning Start')
        for learning_rate in learning_rates:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(epochs):
                count = 0
                print(epoch)
                for data in fairy_data_loader:
                    optimizer.zero_grad()
                    data = torch.stack(data) # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
                    data= data.transpose(1,0)
                    
                    data= data.to(ctx)
            
                    outputs = model(data, labels=data)
                    loss, logits = outputs[:2]
                    loss.backward()
                    optimizer.step()
                    if count %10 ==0:
                        print('epoch no.{} train no.{}  loss = {}' . format(epoch, count+1, loss))
                        # torch.save(model,save_path+'checkpoint_{}_{}.tar'.format(epoch,count))
                        # 추론 및 학습 재개를 위한 일반 체크포인트 저장하기

                    count += 1
            print("save!") # learning_rate 바뀔때마다 저장
            save_path = './checkpoint/'       
            torch.save({
                    'epoch': epoch,
                    'train_no': count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss':loss
                }, save_path+'gt_checkpoint_'+str(learning_rate)+'1.tar')

    else : # 텍스트 생성
        model.eval()
        revers_vocab = {v:k for k,v in vocab.items()}
        text = '나뭇꾼 : '
        input_ids = vocab_b_obj.encode(text)
        gen_ids = model.generate(torch.tensor([input_ids]),
                                max_length=128,
                                repetition_penalty=2.0,
                                pad_token_id=vocab_b_obj.pad_token_id,
                                eos_token_id=vocab_b_obj.eos_token_id,
                                bos_token_id=vocab_b_obj.bos_token_id,
                                use_cache=True)
        generated = vocab_b_obj.decode(gen_ids[0,:].tolist())
        print(generated)


with open('/content/drive/MyDrive/textG/cfg/config_json.json') as f:
    cfg = json.load(f)

ctx= 'cuda'#'cuda' #'cpu' #학습 Device CPU or GPU. colab의 경우 GPU 사용
cachedir = cfg['cachedir'] # KoGPT-2 모델 다운로드 경로

use_cuda = True # Colab내 GPU 사용을 위한 값


main(cfg)


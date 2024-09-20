import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import json
import torch

""" M2M100ForConditionalGeneration : M2M100 모델을 사용, 조건부 생성 작업을 수행하는 모델 클래스
	M2M100Tokenizer : M2M100 모델에 맞는 토크나이저 클래스. 텍스트를 토큰으로 변환하거나 토큰을 텍스트로 복원하는 데 사용
	Seq2SeqTrainer : Seq2Seq 모델의 학습을 위한 Trainer 클래스. 모델 학습 및 평가를 수행.
	Seq2SeqTrainingArguments : Seq2Seq 모델 학습에 필요한 하이퍼파라미터와 설정을 정의하는 클래스. 학습률, 배치 사이즈, 평가 전략 등 다양한 학습 매개변수를 설정. """
from transformers import (M2M100ForConditionalGeneration, M2M100Tokenizer,
Seq2SeqTrainer, Seq2SeqTrainingArguments)
from datasets import Dataset

""" nltk : Natural Language Toolkit의 약자로, 자연어 처리 작업을 위한 다양한 도구와 리소스를 제공. 
    토큰화, 태깅, 파싱, 의미론적 분석 등 다양한 NLP 작업에 사용 """
import nltk

""" sentence_bleu : BLEU (Bilingual Evaluation Understudy) 점수를 계산하는 함수
	SmoothingFunction : BLEU 점수를 계산할 때, 희소성을 해결하기 위해 사용되는 스무딩 기법을 제공하는 클래스. 
						BLEU 점수를 계산할 때 자주 등장하지 않는 단어나 구문이 있어도 점수가 너무 낮게 나오지 않도록 조정. """
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# meteor_score : METEOR (Metric for Evaluation of Translation with Explicit ORdering) 점수를 계산하는 함수
from nltk.translate.meteor_score import meteor_score

# Download wordnet resource
""" NLTK에서 wordnet 리소스 다운로드
	wordNet은 영어 단어들 간의 의미적 관계를 제공하는 사전, NLP 작업에서 의미론적 분석을 수행. 
	METEOR 점수 계산 등에서 필요 """
nltk.download('wordnet')







class M2M:
    """M2M100"""
    def __init__(self, ft_fold, src_lang:str="ko", tgt_lang:str="en"):
        """M2M100 model and tokenizer initialization"""
		""" PyTorch를 사용하여 현재 사용 가능한 하드웨어(CUDA 지원 GPU가 있는 경우 GPU, 그렇지 않으면 CPU)를 선택.
			torch.cuda.is_available() 함수는 CUDA 사용 가능 여부 확인. 이 함수가 True를 반환하면, GPU가 사용 가능. """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# M2M100ForConditionalGeneration 모델을 사전 학습된 가중치로부터 불러오고, 이를 설정한 디바이스(GPU 또는 CPU)에 올림
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(self.device)        

		#  M2M100 모델에 맞는 토크나이저(M2M100Tokenizer)를 로드
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

		# 원본 언어
        self.src_lang = src_lang

		# 타겟 언어
        self.tgt_lang = tgt_lang

		# 토크나이저 원본 언어
        self.tokenizer.src_lang = src_lang

		# 토크나이저 타겟 언어
        self.tokenizer.tgt_lang = tgt_lang

    def trans(self, input_text:str):
        """Translate input_text from source language to target language"""
		# 입력된 텍스트(input_text)를 토크나이저를 통해 텐서 형태로 변환하고, 이를 설정된 디바이스(GPU/CPU)에 맞게 올림
        encoded_pt = self.tokenizer(input_text, return_tensors="pt").to(self.device)

		# 인코딩된 입력을 바탕으로 모델이 번역된 텍스트 토큰 생성. forced_bos_token_id를 사용하여 타겟 언어 지정
        generated_tokens = self.model.generate(**encoded_pt, 
            forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang))

		# 생성된 토큰을 다시 텍스트 형태로 디코딩, 그 결과 반환
        output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return output[0]
        
    def run_trans(self, data_dict: dict, before_after:str="before"):
        """Run translation of source text"""
		# data_dict에서 "source" 키를 통해 번역할 텍스트 목록을 가져오기
        d_source_list = data_dict["source"]
        output_text_list = []
        for idx, input_text in enumerate(d_source_list):
			# 각 텍스트 번역
            output_text = self.trans(input_text)

			# 번역 결과 output_text_list에 저장
            output_text_list.append(output_text)
        k = "trans_" + before_after
        data_dict[k] = output_text_list
    
	# d_source_list의 각 텍스트를 번역 후, 원본과 번역본을 차례대로 출력
    def print_trans(self, d_source_list: list[str]):
        """Print translation of source text"""
        for idx, input_text in enumerate(d_source_list):
            output_text = self.trans(input_text)
            print(f"{idx+1}. {input_text}\n ==>\n{output_text}")

	# 데이터(원본 및 타겟 텍스트)를 모델 학습에 사용할 수 있는 형태로 인코딩
    def encode(self, data):
        # Assuming tokenizer is already initialized and configured
		# 텍스트가 max_length 보다 길면 잘라내기(truncation=True)
        inputs = self.tokenizer(data['source'], padding="max_length", truncation=True, max_length=256)
        outputs = self.tokenizer(data['target'], padding="max_length", truncation=True, max_length=256)
        return {
			# input_ids : 인코딩된 텍스트
            'input_ids': inputs['input_ids'], 

			# attention_mask : 어떤 부분이 실제 텍스트인지 나타내는 마스크
            'attention_mask': inputs['attention_mask'],

			# input_ids : 학습 시, 정답 레이블로 사용
            'labels': outputs['input_ids']  # using output input_ids as labels for training
        }

	# data_dict를 학습 및 테스트 테이터로 분할 후, 인코딩하여 모델에 사용할 수 있는 형태로 변환
    def tokenize(self, data_dict: dict):
        raw_datasets = Dataset.from_dict(data_dict)
        split_datasets = raw_datasets.train_test_split(test_size=0.2)  # 80% train, 20% test
        tokenized_datasets = split_datasets.map(self.encode, batched=True)   
        return tokenized_datasets

def trans(model, tokenizer, input_text:str, src_lang:str="ko", tgt_lang:str="en"):
    """Translate input_text from source language to target language"""
    tokenizer.src_lang = src_lang

    """ PyTorch를 사용하여 현재 사용 가능한 하드웨어(CUDA 지원 GPU가 있는 경우 GPU, 그렇지 않으면 CPU)를 선택.
    	torch.cuda.is_available() 함수는 CUDA 사용 가능 여부 확인. 이 함수가 True를 반환하면, GPU가 사용 가능.(GPU/CPU) """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# 사전 학습된 번역 모델을 설정한 디바이스(GPU/CPU)로 이동
    model.to(device)

	""" return_tensors="pt" 옵션을 사용하여 변환된 데이터를 PyTorch 텐서 형태로 반환
        모델과 입력 데이터가 동일한 디바이스에 있어야 하므로, 토큰화된 데이터도 동일한 디바이스(GPU/CPU)로 전송 """
    encoded_pt = tokenizer(input_text, return_tensors="pt").to(device)    

	""" **encoded_pt: 인코딩된 입력 데이터를 언패킹하여 generate 함수에 전달
		forced_bos_token_id: 생성된 토큰 시퀀스의 시작을 특정 언어로 강제 지정
		tokenizer.get_lang_id(tgt_lang)를 통해 목표 언어의 토큰 ID를 가져옴
		모델이 지정된 목표 언어로 번역을 시작하도록 유도 """
    generated_tokens = model.generate(**encoded_pt, 
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))

	""" 모델이 생성한 토큰 시퀀스를 다시 사람이 읽을 수 있는 자연어 텍스트로 변환
		batch_decode : 여러 시퀀스의 토큰을 동시에 디코딩 
		skip_special_tokens=True : 디코딩 시 특수 토큰(예: <s>, </s>, <pad> 등)을 제거 """
    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return output[0]




	# 파일 로드
def load_data(data_file:str = 'ft_data/data_m2m.json'):
    """Load data from data_m2m.json"""
    with open(data_file, "r", encoding="utf-8") as f:
        data_dict = json.load(f)
    return data_dict

	# 딕셔너리에 있는 원본 텍스트와 타겟 텍스트 출력
def print_data_dict(data_dict: dict):
    for idx, (input_text, output_text) in enumerate(zip(data_dict["source"], data_dict["target"])):
        print(f"Data {idx}\n{input_text}\n ==>\n{output_text}")

	""" m2m : M2M 클래스의 인스턴스
		data_dict : 파인튜닝에 사용할 데이터셋이 포함된 딕셔너리
		ft_fold : 파인튜닝된 모델을 저장할 디렉토리 경로 """
def finetune(m2m:M2M, data_dict:dict, ft_fold:str):

	# tokenize 메소드를 사용하여 데이터셋을 토큰화
    tokenized_datasets = m2m.tokenize(data_dict)

	""" Seq2SeqTrainingArguments 객체를 생성하여 학습 관련 파라미터를 설정
		학습률, 배치 크기, 학습 횟수, 저장 제한 등 다양한 설정이 포함 """
    training_args = Seq2SeqTrainingArguments(
        output_dir=ft_fold,           # Where to store the final model
        evaluation_strategy='epoch',  # Evaluation is done at the end of each epoch
        learning_rate=5e-5,
        fp16=True,
        per_device_train_batch_size=4, # reduce from 16 to 8
        per_device_eval_batch_size=4, # reduce from 16 to 8
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        report_to="none"
    )

	""" Seq2SeqTrainer 객체를 생성하여 학습을 관리
		모델, 학습 파라미터, 학습/평가 데이터셋, 토크나이저 등을 설정 """
    trainer = Seq2SeqTrainer(
        model=m2m.model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=m2m.tokenizer
    )

	# trainer.train()을 호출하여 파인튜닝 시작
    trainer.train()

	# 학습이 완료되면 모델을 지정된 폴더에 저장
    trainer.save_model(ft_fold)

	""" BLEU(Bilingual Evaluation Understudy) 점수 계산
		ref_list: 참조(reference) 문장 리스트
		cand_str: 후보(candidate) 문장 """
def calc_bleu(ref_list:list[str], cand_str:str):
    # Splits to tokens
    references = []

	# 참조 문장 리스트(ref_list)의 각 문장을 단어 토큰으로 분할하여 리스트에 저장
    for t in ref_list:
        t = t.split()
        references.append(t)
        
    # Candidate sentence
	# 후보 문장(cand_str)도 동일하게 단어 토큰으로 분할
    candidate = cand_str.split()
    
    # Calculate BLEU score
	# sentence_bleu 함수를 사용하여 BLEU 점수 계산
    smoothing_function = SmoothingFunction().method1

	# smoothing_function은 점수 계산 시, 스무딩(smoothing)을 적용하기 위한 함수
    bleu_score = sentence_bleu(references, candidate, smoothing_function=smoothing_function)

    return bleu_score    

	""" METEOR(Metric for Evaluation of Translation with Explicit ORdering) 점수 계산
		ref_list: 참조(reference) 문장
		cand_str: 후보(candidate) 문장 """
def calc_meteor(ref_list:list[str], cand_str:str):
    # Splits to tokens
    references = []

	# 참조 문장 리스트(ref_list)의 각 문장을 단어 토큰으로 분할하여 리스트에 저장
    for t in ref_list:
        t = t.split()
        references.append(t)
        
    # Candidate sentence
	# 후보 문장(cand_str)도 동일하게 단어 토큰으로 분할
    candidate = cand_str.split()
    
    # Calculate METEOR score
	# meteor_score 함수를 사용하여 METEOR 점수를 계산
    meteor_sc = meteor_score(references, candidate)

    return meteor_sc

	""" 데이터셋에 대해 지정된 평가 지표(BLEU 또는 METEOR)로 번역 품질 평가
		data_dict: 평가할 데이터가 담긴 딕셔너리
		calc_bleu: 평가에 사용할 함수로 기본값은 calc_bleu
				   필요에 따라 다른 평가 함수로 변경 가능
		metric_name: 평가	"""
def get_score_inplace(data_dict, calc_bleu=calc_bleu, metric_name="BLEU"):
    print(f"Evaluate by {metric_name} score")
    bleu_score_before_list = []
    bleu_score_after_list = []
    for idx in range(len(data_dict["source"])):
        print("Data at ", idx)
        print("Source: ", data_dict["source"][idx])
        print("Target: ", [data_dict["target"][idx]])
        print("Trans before FT: ", data_dict["trans_before"][idx])
        print("Trans after FT: ", data_dict["trans_after"][idx])    
        bleu_score_before = calc_bleu([data_dict["target"][idx]],
                                    data_dict["trans_before"][idx])
        bleu_score_after = calc_bleu([data_dict["target"][idx]],
                                    data_dict["trans_after"][idx])
        bleu_score_after_list.append(bleu_score_after)
        bleu_score_before_list.append(bleu_score_before)
        print(f"{metric_name} score at {idx}: {bleu_score_before} -> {bleu_score_after}")
    data_dict[f"{metric_name} BEFORE"] = bleu_score_before_list
    data_dict[f"{metric_name} AFTER"] = bleu_score_after_list


	""" 주어진 data_dict의 "source" 텍스트를 번역하고, 결과를 출력하며, 필요 시 번역된 텍스트를 data_dict에 추가
		model : 번역에 사용할 M2M100 모델 객체
		tokenizer : 해당 모델에 맞는 토크나이저 객체
		data_dict : "source" 텍스트가 포함된 데이터 딕셔너리
		inplace: 번역된 텍스트를 data_dict에 저장할지 여부를 결정. 기본값은 True	"""
def print_data_dict_trans(model, tokenizer, data_dict, inplace=True):
    output_text_list = []
	
	# data_dict의 "source" 리스트에 있는 각 텍스트를 반복하면서 번역
    for input_text in data_dict["source"]:
        output_text = trans(model, tokenizer, input_text)
        output_text_list.append(output_text)

		# 번역된 텍스트는 output_text_list에 저장하고, 원본 텍스트와 번역 결과를 화면에 출력
        print(f"{input_text}\n\t ==> {output_text}")

	# inplace가 True로 설정된 경우, 번역된 텍스트를 data_dict의 "trans_after" 키에 추가
    if inplace:
        data_dict["trans_after"] = output_text_list

	""" 여러 줄로 구성된 텍스트를 처리하여 단일 라인별로 분리하고, 이를 기반으로 새로운 데이터셋을 생성 """
def load_data_1line():
	# load_data 함수를 사용하여 원본 데이터를 로드
    org_ft_data_dict = load_data()

    # %% convert single line to list
    print("Convert single line to list")

	# 원본 데이터의 "source"와 "target" 리스트에 있는 텍스트의 길이를 출력
    print("Length of original data: ", len(org_ft_data_dict["source"]))

	# 새로운 딕셔너리 ft_data_dict를 생성하여, 이를 기반으로 단일 라인별로 분리된 데이터를 저장
    ft_data_dict = {"source":[], "target":[]}
    for i, (s_line, t_line) in enumerate(zip(org_ft_data_dict["source"], org_ft_data_dict["target"])):

		# "source"와 "target"에서 각각의 텍스트를 "\n\n"로 분리하여 다중 라인을 단일 라인으로 나누고, 이를 ft_data_dict에 저장
        s_lines = s_line.split("\n\n")
        t_lines = t_line.split("\n\n")

		# 분리된 데이터가 같은 길이를 가지는지 확인하고, 만약 길이가 다르면 에러 메시지를 출력	
        if len(s_lines) == len(t_lines):
            ft_data_dict['source'].extend(s_lines)
            ft_data_dict['target'].extend(t_lines)
        else:
            print(f"Error at {i}: {len(s_lines), len(t_lines)}")
            print(s_lines)
            print(t_lines)

	# 변환된 데이터셋의 길이를 출력하고, 최종적으로 새롭게 변환된 ft_data_dict를 반환
    print("Length of converted data (source): ", len(ft_data_dict["source"]))
    print("Length of converted data (target): ", len(ft_data_dict["target"]))   
    
    return ft_data_dict 




# %%
# 모델을 파인튜닝한 결과를 저장할 디렉터리 경로 설정
FT_FOLD = './ft_fold'
# N_FT_DATA = 1600
N_DATA = 30

# 원본 데이터를 불러와서 다중 라인을 단일 라인으로 분리한 후 org_data_dict에 저장
org_data_dict = load_data_1line()

# 파인튜닝에 사용할 데이터셋을 초기화
ft_data_dict = {}

# 원본 데이터에서 N_DATA만큼을 제외한 나머지 데이터를 파인튜닝 데이터셋에 추가
for key in org_data_dict:
    ft_data_dict[key] = org_data_dict[key][:-N_DATA]

# 평가에 사용할 데이터셋을 초기화
eval_data_dict = {}

# 원본 데이터에서 마지막 N_DATA 만큼을 평가 데이터셋에 추가
for key in org_data_dict:
    eval_data_dict[key] = org_data_dict[key][-N_DATA:]


# M2M 클래스의 객체를 생성하여 초기화
m2m = M2M(FT_FOLD)

print("\nBefore finetuning")

# 평가 데이터셋의 소스 텍스트를 번역하고 출력
m2m.print_trans(eval_data_dict["source"])

# 평가 데이터셋의 번역 결과를 "before" 상태로 저장
m2m.run_trans(eval_data_dict, "before")

print("\nTraining data")
print_data_dict(eval_data_dict)

# %%
print(f"\nFine-tuning with {len(ft_data_dict['source'])} data")

# 파인튜닝 수행
finetune(m2m, ft_data_dict, FT_FOLD)

""" 파인튜닝된 모델을 사용하여 평가 데이터를 번역하고, 번역 결과 출력. 
	inplace=False로 설정하여 원본 데이터는 미수정 """
print_data_dict_trans(m2m.model, m2m.tokenizer, eval_data_dict, inplace=False)
# get_score_inplace(data_dict, calc_bleu, "BLEU")
# get_score_inplace(data_dict, calc_meteor, "METEOR")

# %%
model_dir = FT_FOLD

""" PyTorch를 사용하여 현재 사용 가능한 하드웨어(CUDA 지원 GPU가 있는 경우 GPU, 그렇지 않으면 CPU)를 선택.
   	torch.cuda.is_available() 함수는 CUDA 사용 가능 여부 확인. 이 함수가 True를 반환하면, GPU가 사용 가능.(GPU/CPU) """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파인튜닝된 모델을 지정된 디렉터리에서 로드하고, 설정된 디바이스(GPU/CPU)로 이동
model = M2M100ForConditionalGeneration.from_pretrained(model_dir).to(device)

# 파인튜닝된 모델에 맞는 토크나이저를 지정된 디렉터리에서 로드
tokenizer = M2M100Tokenizer.from_pretrained(model_dir)

print("After Fine-tuning")

""" 파인튜닝된 모델을 사용하여 평가 데이터를 번역하고, 번역 결과를 출력
	inplace=True로 설정하여 번역 결과를 평가 데이터셋에 저장 """
print_data_dict_trans(model, tokenizer, eval_data_dict, inplace=True)

# BLEU 점수를 계산하여 출력하는 함수 호출. 파인튜닝 전후의 번역 성능을 비교
get_score_inplace(eval_data_dict, calc_bleu, "BLEU")

# METEOR 점수를 계산하여 출력하는 함수 호출. 파인튜닝 전후의 번역 성능을 비교
get_score_inplace(eval_data_dict, calc_meteor, "METEOR")

# 평가 데이터셋에 대한 번역 결과를 JSON 파일로 저장. 한글이 깨지지 않도록 ensure_ascii=False로 설정.
with open('ft_data/data_m2m_trans_1line.json', "w", encoding="utf-8") as f:
    json.dump(eval_data_dict, f, ensure_ascii=False, indent=4)
print("Data saved!")






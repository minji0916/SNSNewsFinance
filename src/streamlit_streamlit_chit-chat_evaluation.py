import os
import sys

import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, GPTJForCausalLM, BartForConditionalGeneration, LlamaForCausalLM, BitsAndBytesConfig, AutoConfig
import random
import numpy as np

# for LLM
from vllm import LLM, SamplingParams # for speed-up of LLM inference
from vllm.lora.request import LoRARequest
import gc
from huggingface_hub import hf_hub_download
from peft import PeftModel

# for MongoDB
import pymongo
import certifi
from datetime import datetime


os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

# setting for MongoDB
ca = certifi.where() # 보안 설정

MONGO_USERNAME = "lhk4862"  # MongoDB 계정 사용자 이름
MONGO_PASSWORD = "mwozcGbsYzeD6NEf"  # MongoDB 계정 비밀번호
CLUSTER_ADDRESS = "small-talk.objdhkl.mongodb.net"  # MongoDB 클러스터 주소
APP_NAME = "small-talk"
DB_NAME = "evaluation_small-talk_db"



# DB 연결 함수
def get_db_collection(chat_model_name):
    COLLECTION_NAME = "QA_LLAMA3_evaluation_history" if chat_model_name == "beomi/Llama-3-Open-Ko-8B" else "QA_GEMMA_evaluation_history" # QA팀 요청용
    client = pymongo.MongoClient(
        f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@{CLUSTER_ADDRESS}/{DB_NAME}?retryWrites=true&w=majority&appName={APP_NAME}",
        tlsCAFile=ca
    )
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection

# 연결 테스트 함수
def test_db_connection():
    try:
        client = pymongo.MongoClient(
            f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@{CLUSTER_ADDRESS}/{DB_NAME}?retryWrites=true&w=majority&appName={APP_NAME}",
            tlsCAFile=ca
        )
        client.server_info()  # MongoDB 서버 정보 가져오기 시도
        print("Successfully connected to MongoDB Atlas!")
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(f"Failed to connect to MongoDB Atlas: {err}")

# 초기 연결 테스트
###test_db_connection()


# additional settings
FILEPATH = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(FILEPATH), '../'))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
TORCH_USE_CUDA_DSA=1


RANDOM_SEED_SETTING = False # True 시도 -> OOM 발생

### Definitions for DB
def save_chat_to_db(chats, collection):
    # chats (list of dict): 저장할 채팅 데이터의 리스트
    if chats:
        collection.insert_many(chats)

def load_chat_from_db(chat_id, collection):
    return collection.find_one({"_id": chat_id})

# 평가 세션 ID 및 평가 시작 시각 초기화
def initialize_evaluation_session(collection):
    # Initialize session state variables if not already initialized
    if 'db_chats_list' not in st.session_state:
        st.session_state.db_chats_list = []
    if 'session_id' not in st.session_state:
        last_evaluation = collection.find_one(sort=[("evaluation_starttime", -1)])
        
        if last_evaluation and last_evaluation.get('session_id') is not None:
            last_session_id = last_evaluation['session_id']
        else:
            last_session_id = 0  # 기본값 설정
        
        st.session_state.session_id = last_session_id + 1
    if 'evaluation_starttime' not in st.session_state:
        st.session_state.evaluation_starttime = datetime.now().strftime("%y%m%d_%H:%M:%S")  # 현재 시각으로 평가 시작 시각 설정



### Definitions for models
def load_chat_model_(model_name, saved_model_path, device, seed_value=42):
    if model_name == "beomi/Llama-3-Open-Ko-8B":            
        # 모델 로드 전에 기존 모델 삭제 및 가비지 컬렉션 실행
        if 'chat_model' in st.session_state:
            del st.session_state.chat_model
            del st.session_state.chat_tokenizer
            clear_unused_memory()
           
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "Llama-3-Open-Ko" in model_name:
            tokenizer.pad_token = tokenizer.eos_token ### important
             
        model = LLM(model=model_name, seed=seed_value, enable_lora=True, tensor_parallel_size=1, \
                trust_remote_code=True, max_num_seqs=1024, \
                max_loras=1, max_lora_rank=64,max_cpu_loras=2, \
                    enforce_eager=True, gpu_memory_utilization=0.9, max_model_len=1024) # qlora 사용하려면 tensor_parallel_size=1 만 가능 &  set enforce_eager only in GPUs of limited memory
    
    
    elif model_name == "google/gemma-1.1-7b-it":            
        # 모델 로드 전에 기존 모델 삭제 및 가비지 컬렉션 실행
        if 'chat_model' in st.session_state:
            del st.session_state.chat_model
            del st.session_state.chat_tokenizer
            clear_unused_memory()
           
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        tokenizer.padding_side = 'right'
        

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
        ) 
        model = PeftModel.from_pretrained(model, saved_model_path)
        
    return model, tokenizer

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        
def clear_unused_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(**st.secrets["mongo"])

@st.cache_resource
def load_chat_model(model_name, saved_model_path, device_number, this_seed_value):
    device = torch.device(f'cuda:{device_number}') if torch.cuda.is_available() else 'cpu'

    # 스트림릿 세션 상태에 모델과 토크나이저 저장 (0626 수정)
    model, tokenizer = load_chat_model_(model_name, saved_model_path, device, seed_value=this_seed_value)
    st.session_state.chat_model_name = model_name
    st.session_state.chat_saved_model_path = saved_model_path
    st.session_state.chat_model = model
    st.session_state.chat_tokenizer = tokenizer

    return model, tokenizer

### for LLAMA-3 모델
def get_input_content(chat, tokenizer):
    # messages: List of dict
    # [{'role': 'user', 'content': '안녕'}, {'role': 'assistant', 'content': ' 안녕하세요. 저는 20대 여성입니다! '}, {'role': 'user', 'content': '반가워요'}]
    history_message = ""
    for turn_dict in chat:
        role, content = turn_dict['role'], turn_dict['content'] # 발화자 구분 (0: 사용자, 1: 시스템)
        print("= = = = 디버깅 = = = =")
        print("= = turn_dict['role']: ", role)
        print("= = = turn_dict['content']: ", content, end="\n")
        
        ### (1) 기존 프롬프트 방식
        # if role == 'user':
        #     if history_message == "": history_message += f"0 :  {content}"
        #     else: history_message += f"\n0 :  {content}"
        # else:
        #     history_message += f"\n1 :  {content}"
        
        ### (2) 변경 프롬프트 방식
        if role == 'user':
            if history_message == "": history_message += f"<|reserved_special_token_0|>{content}"
            else: history_message += f"<|reserved_special_token_0|>{content}"
        else:
            history_message += f"<|reserved_special_token_1|>{content}"
    
    ### (1) 기존 프롬프트 방식        
    # history_message += "\n1 : "
    ### (2) 변경 프롬프트 방식
    history_message += "<|reserved_special_token_1|>" ### 0625 수정
    prompt = history_message
    tokenized_history = tokenizer.encode(
                                        prompt,
                                        add_special_tokens=False ### 0614 추가 완료
                                        )


    
    # trunctation
    if tokenizer.name_or_path == "beomi/Llama-3-Open-Ko-8B":
        input_ids, attention_mask = None, None
        cutoff_len = (1024 - 128)
        history_ids = tokenized_history
        if len(history_ids) >= cutoff_len:
            history_ids = history_ids[-(cutoff_len):] # left side trunctate
        
        context_text = str(tokenizer.decode(history_ids))


    result = {'input_encoded': input_ids,
                'mask_encoded': attention_mask,
                'context_text': context_text
                }

    return prompt, result['input_encoded'], result['mask_encoded'], context_text


def get_output_content(chat, model, tokenizer, saved_model_path):
    prompt, input_encoded, mask_encoded, context_text = get_input_content(chat, tokenizer)
    
    ### model generate
    # (2) for Sparta-large model
    if tokenizer.name_or_path == "beomi/Llama-3-Open-Ko-8B": # isinstance(model, LlamaForCausalLM): -> X (model is a object of vLLM)
        use_beam_search = False
        if use_beam_search:
            # best_of만 custom 가능, 나머지는 고정
            best_of = 4
            temperature = 0.0
            top_k = -1
            top_p = 1.00
        else:
            # best_of는 무조건 1이고, 나머지는 custom 가능
            best_of = 1
            temperature = 0.1 ### 0617 변경: 0.0 -> 0.1
            top_k = 50
            top_p = 0.1
            
        sampling_params = SamplingParams( # # for Sparta-large model
            temperature=temperature,
            use_beam_search=use_beam_search,
            best_of=best_of,
            top_k=top_k,
            top_p=top_p,
            skip_special_tokens=True,
            
            stop=[tokenizer.eos_token, "\n", "<|reserved_special_token_0|>", "<|reserved_special_token_1|>"], # 예: '<|endoftext|>'
            stop_token_ids=[tokenizer.eos_token_id],
            
            repetition_penalty=1.3,
            
            ### 기존 no_repeat_ngram_size=4 -> vLLM: no argument named no_repeat_ngram_size in SamplingParams
            presence_penalty=1.3,
            frequency_penalty=1.3,
            
            max_tokens=50,
            
            ### Add for Lora with Quantization Inference
            ### reference: https://vllm.readthedocs.io/_/downloads/en/stable/pdf/
            logprobs=1,
            prompt_logprobs=1,
        )

        # print("(디버깅) context_text:", context_text)
        output = model.generate([context_text], sampling_params, use_tqdm=False,
                                    lora_request=LoRARequest("chit-chat_adapter", 1, saved_model_path) # To use LoRA adapter, set LoRARequest
                                    ) # output -> List
        output_text = output[0].outputs[0].text
        
        
        # print("(디버깅) output_text:", output[0].outputs)
        # print(type(output[0].outputs), len(output[0].outputs))

    


    if isinstance(model, GPTJForCausalLM):
        return_text = output_text[len(prompt):]    
    elif isinstance(model, BartForConditionalGeneration) or tokenizer.name_or_path == "beomi/Llama-3-Open-Ko-8B":
        return_text = output_text
    return return_text


### for GEMMA 모델
def gen(text, model, tokenizer):
    result = model.generate(
        **tokenizer(text,
                    return_tensors='pt',
                    add_special_tokens=False,
                    return_token_type_ids=False).to("cuda"),
        max_new_tokens=100,
        eos_token_id=tokenizer.eos_token_id)
    return result


def main():
    st.title("Multiturn Chit-Chat")

    model_options = {
        'Sparta-large (LLAMA-3)': ('beomi/Llama-3-Open-Ko-8B', '/home/tmax/hyunkyung_lee/multi-turn/open-domain_multiturn/HF_Trainer/T-BART_fully_finetuning/LLAMA-3_100%_lora-rank_64_unused_token_runtest_10k/checkpoint-10000', 0),
        
        ### 추가
        'Sparta-large (GEMMA)': ('google/gemma-1.1-7b-it', '/home/tmax/hyunkyung_lee/multi-turn/open-domain_multiturn/HF_Trainer/bowon_gemma/outputs/gemma/7b/checkpoint', 1)
        
    }
    


    with st.sidebar:
        model_choice = st.radio("Select Model", list(model_options.keys()), index=0)
        clear_chat = st.button('Clear Chat')
        eval_mode = st.checkbox('평가 모드 활성화')
        
        evaluator_name = ""
        if eval_mode:
            evaluator_name = st.text_input('평가자 이름을 입력하세요.')
            
    model_name, saved_model_path, device_number = model_options[model_choice]

    # seed setting
    if RANDOM_SEED_SETTING:
        random_seed_value = random.randint(0, 10000)
    else:
        random_seed_value = 42 # 42
        
    set_random_seed(random_seed_value)
    
    
    if 'chat_model' not in st.session_state:
        st.session_state.chat_model_name = model_name
        st.session_state.chat_saved_model_path = saved_model_path
        with st.spinner("채팅 모델 로딩 중..."):
            st.session_state.chat_model, st.session_state.chat_tokenizer = load_chat_model(model_name=st.session_state.chat_model_name, saved_model_path=st.session_state.chat_saved_model_path, device_number=device_number, this_seed_value=random_seed_value)



    if clear_chat:
        # 모델 로드 전에 기존 모델 삭제 및 가비지 컬렉션 실행
        del st.session_state.chat_model
        del st.session_state.chat_tokenizer
        clear_unused_memory()
        st.session_state.chat_model, st.session_state.chat_tokenizer = load_chat_model(
            model_name=st.session_state.chat_model_name, 
            saved_model_path=st.session_state.chat_saved_model_path, 
            device_number=device_number, 
            this_seed_value=random_seed_value
        )
        if eval_mode:
            collection = get_db_collection(st.session_state.chat_model_name)
            initialize_evaluation_session(collection)



    if ('messages_2' not in st.session_state) or clear_chat:
        st.session_state.messages_2 = []
    if ('messages_3' not in st.session_state) or clear_chat:
        st.session_state.messages_3 = []
    if ('messages_4' not in st.session_state) or clear_chat:
        st.session_state.messages_4 = []

    if 'current_model' not in st.session_state or st.session_state.current_model != model_choice:
        st.session_state.current_model = model_choice
        with st.spinner(f"Loading {model_choice}..."):
            st.session_state.chat_model, st.session_state.chat_tokenizer = load_chat_model(model_name, saved_model_path, device_number, this_seed_value=random_seed_value)

    if st.session_state.current_model == 'Sparta-small (GPT-J)':
        messages = st.session_state.messages_2
    elif st.session_state.current_model == 'Sparta-large (GEMMA)':
        messages = st.session_state.messages_3
    elif st.session_state.current_model == 'Sparta-large (LLAMA-3)':
        messages = st.session_state.messages_4

    for message in messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    chat_instruction = 'Chat with Sparta model!'


    # 평가자 이름 설정
    if eval_mode and evaluator_name:
        evaluator_name = evaluator_name.strip()
        if evaluator_name:
            st.session_state.evaluator_name = evaluator_name

    evaluator_name = st.session_state.get('evaluator_name', 'Unknown')
    # st.write('' if evaluator_name=='Unknown' else f"현재 평가자: {evaluator_name}")
    
    

    
    if eval_mode:
        collection = get_db_collection(st.session_state.chat_model_name)
        initialize_evaluation_session(collection)

    
    evaluation_starttime = st.session_state.get('evaluation_starttime', None)
    session_id = st.session_state.get('session_id', None)


    if prompt := st.chat_input(chat_instruction):
        with st.chat_message('user'):
            st.markdown(prompt)

        if clear_chat:
            if st.session_state.current_model == 'Sparta-small (GPT-J)':
                st.session_state.messages_2 = []
            elif st.session_state.current_model == 'Sparta-large (GEMMA)':
                st.session_state.messages_3 = []
            elif st.session_state.current_model == 'Sparta-large (LLAMA-3)':
                st.session_state.messages_4 = []
            st.session_state.db_chats_list = []  # DB 저장 리스트도 초기화

        if st.session_state.current_model == 'Sparta-small (GPT-J)':
            st.session_state.messages_2.append({'role': 'user', 'content': prompt})
        elif st.session_state.current_model == 'Sparta-large (GEMMA)':
            st.session_state.messages_3.append({'role': 'user', 'content': prompt})
        elif st.session_state.current_model == 'Sparta-large (LLAMA-3)':
            st.session_state.messages_4.append({'role': 'user', 'content': prompt})

        messages = (
                    st.session_state.messages_2 if st.session_state.current_model == 'Sparta-small (GPT-J)' else
                    st.session_state.messages_3 if st.session_state.current_model == 'Sparta-large (GEMMA)' else
                    st.session_state.messages_4
                    )        # messages: List of dict
        # [{'role': 'user', 'content': '안녕'}, {'role': 'assistant', 'content': ' 안녕하세요. 저는 20대 여성입니다! '}, {'role': 'user', 'content': '반가워요'}]
        # print("get_output_content 함수에 들어가는 messages: ", messages)
        # if messages[-1]['role'] != 'user':
        #     print(f"ERROR !!! (messages: {messages})")
        
        
        if st.session_state.current_model == 'Sparta-large (LLAMA-3)':
            response = get_output_content(
                messages, 
                st.session_state.chat_model, 
                st.session_state.chat_tokenizer,
                ###device=f'cuda:{device_number}'
                saved_model_path
            )
            
        elif st.session_state.current_model == 'Sparta-large (GEMMA)':    
            input_prompt = st.session_state.chat_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = gen(input_prompt, st.session_state.chat_model, st.session_state.chat_tokenizer)
            result = st.session_state.chat_tokenizer.decode(outputs[0]).split("<start_of_turn>model")[-1]
            response = result.split("<end_of_turn>")[0].strip()
 
            
            
        if eval_mode:
            evaluation_starttime = datetime.now().strftime("%y%m%d_%H:%M:%S")  # 현재 시각으로 평가 시작 시각 업데이트
            this_turn_id = (len(messages) // 2 + 1)

            db_instance_row = {
                                'session_id': session_id, # 세션 ID (int)
                                'evaluator_name': evaluator_name, # 평가자 이름 (str)
                                'evaluation_starttime': evaluation_starttime, # 평가 시작 일자 및 시각 (str)
                                
                                'sparta_model': st.session_state.chat_model_name,
                                    
                                'turn_id': this_turn_id, # 세션 내 턴 번호 (int)
                                'user_utterance': messages[-1]['content'], # 사용자 입력 발화 (str)
                                'system_response': response # str # 모델 생성 응답 (str)
                            }
            
            st.session_state.db_chats_list.append(db_instance_row)
        
        
        with st.chat_message('assistant'):
            st.markdown(response)


        if st.session_state.current_model == 'Sparta-small (GPT-J)':
            st.session_state.messages_2.append({'role': 'assistant', 'content': response})
        elif st.session_state.current_model == 'Sparta-large (GEMMA)':
            st.session_state.messages_3.append({'role': 'assistant', 'content': response})
        elif st.session_state.current_model == 'Sparta-large (LLAMA-3)':
            st.session_state.messages_4.append({'role': 'assistant', 'content': response})


    # "DB에 저장하기" 버튼 추가
    if st.button('대화 저장하기'):
        if eval_mode and st.session_state.db_chats_list:
            collection = get_db_collection(st.session_state.chat_model_name) ### 여기 한줄 더 추가 (0628)
            save_chat_to_db(st.session_state.db_chats_list, collection)
            st.session_state.db_chats_list = []  # 저장 후 리스트 초기화
            if evaluator_name!='Unknown':
                st.write(f"현재 평가자: {evaluator_name}")
            if evaluation_starttime:
                st.write(f"평가 시작 시각: {st.session_state.evaluation_starttime}")

            st.success("채팅 기록이 성공적으로 저장되었습니다.")
            initialize_evaluation_session(collection)
        else:
            st.warning("저장할 채팅 기록이 없습니다.")

if __name__ == '__main__':
    main()
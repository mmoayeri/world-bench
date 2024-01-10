from abc import ABC, abstractmethod
from torch import Tensor
from constants import _CACHED_DATA_ROOT, _API_KEYS, _LLM_CACHE_DIR
from my_utils import cache_data, load_cached_data
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import load_model, get_conversation_template, add_model_args
import torch
import os
from tqdm import tqdm

class LLM(ABC):

    def __init__(self):
        raise NotImplementedError

    def get_modelname(self) -> str:
        return self.model_key

    @abstractmethod
    def answer_questions(self, questions: List[str]) -> List[str]:
        raise NotImplementedError

    # @abstractmethod
    # def parse_answer(self, answer: str) -> List[str]:
    #     # given single answer to a prompt, separate all individual attributes contained in the answer
    #     # E.g. asking 'list diff kinds of fox' may yield '1. Kit fox\n2. Arctic fox\n3. Red fox'
    #     # this function converts that answer string to ['Kit fox', 'Arctic fox', 'Red fox']
    #     raise NotImplementedError

# class Vicuna(LLM):

#     def __init__(self, model_key: str ='vicuna-13b-v1.5'):
#         self.model_key = model_key
#         # Loading the LLM takes a lot of space, so we don't unless we need to. 
#         self.model = 'NOT YET LOADED'
#         self.batch_size = 32

#     def set_up_model(self):
#         self.model, self.tokenizer = load_model(f'lmsys/{self.model_key}', device='cuda', num_gpus=1)
#         self.tokenizer.padding_side = 'left'
#         # First, we provide a general initial prompt to the LLM.
#         conv = get_conversation_template(f'lmsys/{self.model_key}')
#         starter_prompt = conv.get_prompt()
#         _ = self.answer_questions([starter_prompt]) 

#         #     generated_ids = model.generate(input_ids, max_new_tokens=256, do_sample=True)
#         #     decoded = tokenizer.decode(generated_ids[0][len(input_ids[0]):])
#         #     outputs.append(decoded)

#         # all_outputs = []
#         # for i in tqdm(range(0, len(questions), self.batch_size)):
#         #     prompt_list = ['User: '+q+'\nAssistant:' for q in questions[i:i+self.batch_size]]
#         #     input_ids = self.tokenizer(prompt_list, padding=True).input_ids
#         #     input_tens = torch.tensor(input_ids).cuda()
#         #     output_ids = self.model.generate(input_tens, do_sample=True, temperature=0.7, repetition_penalty=1, max_new_tokens=256)
#         #     outputs = [self.tokenizer.decode(out_ids[len(in_ids):], skip_special_tokens=True, spaces_between_special_tokens=False) 
#         #                 for in_ids, out_ids in zip(input_ids, output_ids)]
#         #     all_outputs.extend(outputs)
#         # return all_outputs

#     def parse_answer(self, answer: str) -> List[str]:
#         '''
#         TODO: We should think more about this...

#         Currently, this expects responses to be in the form of a numbered list. 
#         So we expect answer to look like "1. Kit fox\n2. Arctic fox\n3. Red fox"
#         '''
#         # separate per line
#         individual_answers = answer.split('\n')
#         # remove leading numbers like '1. '
#         # sometimes there is a period at the end of a response, we remove that as well
#         # sometimes the llm says {axes of variance}: {instances} (e.g. Appearance: decorative); we just want the later part
#         cleaned_answers = [ans.split('. ')[-1].split(':')[-1].strip().replace('.', '') for ans in individual_answers]
#         return cleaned_answers

class HuggingFaceLLM(LLM):
    def __init__(self, hf_model_id: str):
        self.hf_model_id = hf_model_id
        self.model = 'NOT YET LOADED'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if '7b' in hf_model_id or 'phi-2' in hf_model_id:
            self.batch_size = 16
        elif 'Qwen' in hf_model_id:
            self.batch_size = 1
        else:
            self.batch_size = 4

    def set_up_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_id, 
            cache_dir=_LLM_CACHE_DIR, 
            trust_remote_code=True,
            padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_id, 
            cache_dir=_LLM_CACHE_DIR, 
            trust_remote_code=True,
            device_map="auto").eval()#.to(self.device)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if not self.tokenizer.eos_token:
                msg =  f"pad_token and eos_token are None, so batchsize must be 1. Batchsize is currently {self.batch_size}." 
                assert self.batch_size == 1, msg

        # self.model = self.model

    def get_modelname(self) -> str:
        return self.hf_model_id.split('/')[-1]

    # def format_and_tokenize(self, base_instruction: str, example: Tuple[str, str], question: str) -> Tensor:
    def format_input(self, base_instruction: str, example: Tuple[str, str], question: str) -> Tensor:
        eg_question, eg_answer = example
        messages = [
            {"role": "user", "content": base_instruction},
            {"role": "assistant", "content": "Sounds good, will do."},
            {"role": "user", "content": eg_question},
            {"role": "assistant", "content": eg_answer},
            {"role": "user", "content": question}
        ]
        # input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        # return input_ids

        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return inputs
        
    def answer_questions(
        self, 
        questions: List[str], 
        base_instruction: str, 
        example: Tuple[str, str]
        ) -> List[str]:
        if self.model == 'NOT YET LOADED':
            self.set_up_model()
        # input_ids = self.format_and_tokenize(base_instruction, example, questions)

        outputs = []
        # for question in tqdm(questions):
        #     input_ids = self.format_and_tokenize(base_instruction, example, question)
        #     output_ids = self.model.generate(input_ids.to(self.device), max_new_tokens=64, do_sample=True, repetition_penalty=1)
        #     decoded = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:])
        #     outputs.append(decoded)

        padding = (self.batch_size > 1)    
        
        for i in tqdm(range(0, len(questions), self.batch_size)):
            with torch.no_grad():
                inputs = [self.format_input(base_instruction, example, question) for question in questions[i:i+self.batch_size]]
                input_ids = self.tokenizer(inputs, padding=padding, return_tensors='pt').input_ids.to(self.device)
                output_ids = self.model.generate(input_ids, max_new_tokens=64, do_sample=True, repetition_penalty=1)
                batch_outputs = self.tokenizer.batch_decode(output_ids[:,input_ids.shape[1]:], skip_special_tokens=True, spaces_between_special_tokens=False)
                outputs.extend(batch_outputs)
                del input_ids; del output_ids; torch.cuda.empty_cache()
        return outputs

class SimpleChatTemplateLLM(HuggingFaceLLM):
    ### for Vicuna and Qwen
    # def format_and_tokenize(self, base_instruction: str, example: Tuple[str, str], question: str) -> Tensor:
    def format_input(self, base_instruction: str, example: Tuple[str, str], question: str) -> Tensor:
        eg_question, eg_answer = example
        prompt = "USER: " + base_instruction + "\n"
        prompt += "ASSISSTANT: Sounds good, will do.\n"
        prompt += "USER: " + eg_question + "\n"
        prompt += "ASSISSTANT: " + eg_answer + "\n"
        prompt += "USER: " + question + "\n"
        prompt += "ASSISSTANT: "

        # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        # return input_ids
        return prompt

class RemoteCodeLLM(HuggingFaceLLM):
    def set_up_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id, cache_dir=_LLM_CACHE_DIR, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_model_id, cache_dir=_LLM_CACHE_DIR, trust_remote_code=True)#.to(self.device)
        self.model = self.model.eval().to(self.device)

class Llama2(HuggingFaceLLM):
    def set_up_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_id, 
            cache_dir=_LLM_CACHE_DIR, 
            token=_API_KEYS["llama"],
            padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_id, 
            cache_dir=_LLM_CACHE_DIR, 
            token=_API_KEYS["llama"],
            device_map='auto').eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token

# class 

### Closed source models accessed via API key

class Cohere(LLM):
    def __init__(self, model_key: str = "command"):
        assert model_key in ["command", "command-light"], f"Invalid model_key {model_key} for cohere LLMs"
        self.cohere_model_key = model_key
        self.model_key = "cohere__"+model_key
        import cohere
        self.co = cohere.Client(_API_KEYS["cohere"])

    def get_modelname(self) -> str:
        return self.model_key

    def answer_questions(
        self, 
        questions: List[str], 
        base_instruction: str, 
        example: Tuple[str, str]
        ) -> List[str]:

        eg_question, eg_answer = example
        chat_history = [
            {"role": "USER", "message": base_instruction},
            {"role": "CHATBOT", "message": "Sounds good, will do."},
            {"role": "USER", "message": eg_question},
            {"role": "CHATBOT", "message": eg_answer},
        ]
        responses = [self.co.chat(model=self.cohere_model_key, chat_history=chat_history, message=q).text for q in tqdm(questions)]
        return responses


class OpenAI(LLM):
    def __init__(self, model_key: str = "gpt-3.5-turbo"):
        assert model_key in ["gpt-3.5-turbo", "gpt-4"], f"Invalid model_key {model_key} for OpenAI LLMs"
        self.model_key = model_key
        from openai import OpenAI
        self.client = OpenAI(api_key=_API_KEYS['openai'])

    def get_modelname(self) -> str:
        return self.model_key

    def answer_questions(
        self, 
        questions: List[str], 
        base_instruction: str, 
        example: Tuple[str, str]
        ) -> List[str]:

        eg_question, eg_answer = example

        responses = []
        for question in tqdm(questions):
            messages = [
                {"role": "user", "content": base_instruction},
                {"role": "assistant", "content": "Sounds good, will do."},
                {"role": "user", "content": eg_question},
                {"role": "assistant", "content": eg_answer},
                {"role": "user", "content": question}
            ]
            response = self.client.chat.completions.create(model=self.model_key, messages = messages, max_tokens=64)
            responses.append(response.choices[0].message.content)
        return responses


_LLM_DICT = dict({
    "zephyr_7b": (HuggingFaceLLM, "HuggingFaceH4/zephyr-7b-beta"),
    "mistral_7b_instruct": (HuggingFaceLLM, "mistralai/Mistral-7B-Instruct-v0.2"),
    # "wizard_7b": (HuggingFaceLLM, "WizardLM/WizardLM-7B-V1.0"), 
    # "wizard_13b": (HuggingFaceLLM, "WizardLM/WizardLM-13B-V1.2"), 
    # "wizard_30b": (HuggingFaceLLM, "WizardLM/WizardLM-30B-V1.2"),
    "vicuna-7b-v1.5": (SimpleChatTemplateLLM, 'lmsys/vicuna-7b-v1.5'),
    # "vicuna-33b-v1.3": (Vicuna, 'lmsys/vicuna-33b-v1.3'),
    # "vicuna-13b-v1.5": (Vicuna, 'vicuna-13b-v1.5'),
    "qwen_7b": (SimpleChatTemplateLLM, "Qwen/Qwen-7B"),  # No chat template, use 'USER: ... ASSISTANT:' convention
    "qwen_7b_chat": (SimpleChatTemplateLLM, "Qwen/Qwen-7B-Chat"), 
    "llama-2_7b": (Llama2, "meta-llama/Llama-2-7b-hf"),
    "llama-2_7b_chat": (Llama2, "meta-llama/Llama-2-7b-chat-hf"),
    "orca_7b": (HuggingFaceLLM, "microsoft/Orca-2-7b"),
    "phi-2": (HuggingFaceLLM, "microsoft/phi-2"),

    "vicuna-13b-v1.5": (SimpleChatTemplateLLM, 'lmsys/vicuna-13b-v1.5'),
    "orca_13b": (HuggingFaceLLM, "microsoft/Orca-2-13b"),
    "llama-2_13b": (Llama2, "meta-llama/Llama-2-13b-hf"),
    "llama-2_13b_chat": (Llama2, "meta-llama/Llama-2-13b-chat-hf"),
    "qwen_14b": (SimpleChatTemplateLLM, "Qwen/Qwen-14B"), 
    "qwen_14b_chat": (SimpleChatTemplateLLM, "Qwen/Qwen-14B-Chat"), 
    # # Closed source / API models
    # "cohere__command": (Cohere, "command"),
    # "gpt-3.5-turbo": (OpenAI, "gpt-3.5-turbo"),
    # "gpt-4": (OpenAI, "gpt-4")
})
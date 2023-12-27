from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import _LLM_CACHE_DIR, _API_KEYS


def download_model_and_tokenizer(hf_model_id: str):
    try:
        if 'llama' in hf_model_id:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, cache_dir=_LLM_CACHE_DIR, token=_API_KEYS["llama"])
            model = AutoModelForCausalLM.from_pretrained(hf_model_id, cache_dir=_LLM_CACHE_DIR, token=_API_KEYS["llama"])
        elif "Qwen" in hf_model_id:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, cache_dir=_LLM_CACHE_DIR, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(hf_model_id, cache_dir=_LLM_CACHE_DIR, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, cache_dir=_LLM_CACHE_DIR)
            model = AutoModelForCausalLM.from_pretrained(hf_model_id, cache_dir=_LLM_CACHE_DIR)
    except:
        pass

def download_all():
    for hf_model_id in [
        "lmsys/vicuna-7b-v1.5", "lmsys/vicuna-13b-v1.5", "lmsys/vicuna-33b-v1.3",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "WizardLM/WizardLM-7B-V1.0", "WizardLM/WizardLM-13B-V1.2", "WizardLM/WizardLM-30B-V1.2",
        "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat",
        ]:
        download_model_and_tokenizer(hf_model_id)

if __name__ == '__main__':
    download_all()
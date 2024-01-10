import pickle
import torch
import numpy as np
import os
from tqdm import tqdm
import json
import pandas as pd
# from llm import _LLM_DICT
# from query import _QUERY_DICT


def remove_outer_axes(f, ax):
    ax = f.add_axes([0,0,1,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return f,ax

def cache_data(cache_path: str, data_to_cache, mode: str):
    os.makedirs('/'.join(cache_path.split('/')[:-1]), exist_ok=True)

    if mode == 'pickle':
        with open(cache_path+'.pkl', 'wb') as f:
            pickle.dump(data_to_cache, f)
    elif mode == 'json':
        with open(cache_path+'.json', 'w') as f:
            json.dump(data_to_cache, f)
    else:
        raise ValueError(f"mode for cache_data must be json or pickle")

def load_cached_data(cache_path: str, mode: str):
    if mode == 'pickle':
        with open(cache_path+'.pkl', 'rb') as f:
            dat = pickle.load(f)
    elif mode == 'json':
        with open(cache_path+'.json', 'r') as f:
            dat = json.load(f)
    else:
        raise ValueError(f"mode for load_cached_data must be json or pickle")
    
    return dat


def parse_number_str(answer_str: str):
    """
    other things to strip: [INST], <s>, </s>, \n
    <|assistant|>\n is sometimes at the beginning
    """

    # let's strip away some things

    # for zephyr
    answer_str = answer_str.replace('<|assistant|>\n', '').split('</s>')[0]

    if answer_str[-8:] == ' people.':
        answer_str = answer_str[:-8]
    if ' ' in answer_str:
        words = answer_str.split(' ')
        if words[-2] == 'in':
            words = words[:-2]
        answer_str = ' '.join(words)
    if answer_str[-1] == '.':
        answer_str = answer_str[:-1]
    if answer_str[-1] == '%':
        answer_str = answer_str[:-1]

    answer_str = answer_str.replace(',', '')

    try:
        # attempt 1: will convert any answers that are already a number, perhaps with commas
        # or a sentence where last word is number
        try:
            number = float(answer_str)
        except:
            words = answer_str.split(' ')
            if words[-2] == 'in': # get rid 
                words = words[:-2]
            number = float(words[-1])
    except:
        # first attempt failed
        number = np.nan
        suffix_to_num = dict({'thousand': 1e3, 'million': 1e6, 'billion': 1e9})
        words = answer_str.lower().split(' ')
        for suffix in suffix_to_num:
            if suffix in words:
                try:
                    index = words.index(suffix)
                    number = float(words[index-1].replace(',', '')) * suffix_to_num[suffix]
                    break
                except:
                    pass
    
    # hard coded error fix: sometimes my parsing code above returns the year the LLM cites it's answer is from. Idk how to fix this.
    if np.abs(number - 2020) < 10:
        number = np.nan

    return number

def parse(answer_str: str):
    answer_str = answer_str.strip()
    if '[/INST] ' in answer_str:
        answer_str = answer_str.split('[/INST] ')[-1]
    if 'correct answer is: ' in answer_str:
        answer_str = answer_str.split('correct answer is: ')[1].split(' ')[0]

    words = answer_str.split(' ')
    for suffix in ['thousand', 'million', 'billion', 'trillion']:
        suffix_to_num = dict({'million': 1e6, 'billion': 1e9, 'trillion': 1e12, 'thousand': 1e3, 'hundred': 1e2})
        for suffix in suffix_to_num:
            if suffix in words:
                ind = words.index(suffix)
                try:
                    answer_str = str(float(words[ind-1]) * suffix_to_num[suffix])
                except:
                    pass

    for prequel_word in ['approximately', 'about']:
        if prequel_word in words:
            ind = words.index(prequel_word)
            if ind + 1 < len(words):
                answer_str = words[ind+1]
                break

    answer_str = answer_str.replace('<|assistant|>\n', '').replace(' [/INST]', '').replace('*', '')
    answer_str = answer_str.replace('<|im_start|>assistant\n', '').split('</s>')[0].replace(',', '').split('\n')[0].split('<|im_end|>')[0]
    
    try: 
        _ = float(answer_str.split(' ')[0].split('\\')[0].split('%')[0])
        answer_str = answer_str.split(' ')[0].split('\\')[0].split('%')[0]
    except: # last attempt, let's take last word
        answer_str = answer_str.split(' ')[-1].split('\\')[0].split('%')[0]

    if len(answer_str) > 0 and answer_str[-1] == '.':
        answer_str = answer_str[:-1]

    try: 
        if np.abs(float(answer_str) - 2020) < 10 or float(answer_str) < 0:
            answer_str = np.nan
    except:
        pass

    # try:
    #     answ

    return answer_str

### Some utils to help write my parser

def count_success(parsed, answers, verbose=False):
    cc, ctr = 0, 0 
    for k, v in list(parsed.items()):
        ctr += 1
        try: 
            _ = float(v)
            cc += 1
        except:
            if verbose:
                print("Answer: ", answers[k])
                print("Parsed: ", parsed[k])
                print()
    return cc/ctr * 100

cats = pd.read_csv('gt_answers/country_categories/region_and_income.csv')
to_skip = [c.replace('\\', '') for c in cats.iloc[220:].Economy]

def try_parsing(query_name, llm, parse_fn, verbose=False):
    # query = _QUERY_DICT[query_name]()
    answers = load_cached_data(os.path.join(_CACHED_DATA_ROOT, 'answers', query_name, llm.get_modelname()), mode='json')['answers']#query.query(llm)
    ans2 = dict({k:parse_fn(v) for k,v in answers.items() if k not in to_skip})
    success_rate = count_success(ans2, answers, verbose=verbose)
    print(f"Query: {query_name:<40}, Success rate: {success_rate:.2f}%")
    return success_rate

def try_all(llm, parse_fn):
    success_rates = []
    for query_name in _QUERY_DICT:
        success_rates.append(try_parsing(query_name, llm, parse_fn))
    avg = np.mean(success_rates)
    print(f'Average Success rate: {avg:.2f}')
    return avg

# def test_parsing_all_llms(parse):
#     avgs = []
#     for llm_name, (make_llm, llm_key) in _LLM_DICT.items():
#         llm = make_llm(llm_key)
#         print(llm_name)
#         avgs.append(try_all(llm, parse))
#         print()
#     print(f'Average over all llms and queries: {np.mean(avgs):.4f}%')


def beautify(s, mode='Region'):
    region_map = dict({
        "Sub-Saharan Africa": "Sub-Saharan\nAfrica", 
        "East Asia \& Pacific": "East Asia\n\& Pacific", 
        "Middle East \& North Africa": "Middle East\n\& North Africa",
        "South Asia": "South\nAsia", 
        "Latin America \& Caribbean": "Latin America\n\& Caribbean",
        "Europe \& Central Asia": "Europe \&\nCentral Asia", 
        "North America": "North\nAmerica"
    })

    income_map = dict({
        "Low income": "Low Income",
        "Lower middle income": "Lower Middle\nIncome",
        "Upper middle income": "Upper Middle\nIncome",
        "High income": "High Income",
    })

    error_type_map = dict({
        "AbsRelError": "Absolute Relative Error",
        "RelError": "Relative Error"
    })

    query_map = dict({
        "population": "Population",
        "unemployment": "Unemployment",
        "maternal_mortality_ratio": "Maternal\nMortality Rate",
        "women_in_parliament": "Women In\nParliament",
        "education_expenditure": "Education\nExpenditure",
        "electricity_access_percent": "Electricity\nAccess",
        "agricultural_land_percent": "Agricultural\nLand Percent",
        "co2_emissions": "CO2\nEmissions",
        "gdp": "GDP",
        "gdp_ppp": "GDP PPP Per\nPerson Employed",
        "renewable_percent": "Renewable\nEnergy Ratio",
    })

    llm_map = dict({
        "zephyr_7b": "Zephyr 7B",
        "mistral_7b_instruct": "Mistral 7B\nInstruct",
        "vicuna-7b-v1.5": "Vicuna 7B\nv1.5",
        "qwen_7b": "Qwen 7B",
        "qwen_7b_chat": "Qwen 7B\nChat",
        "llama-2_7b": "Llama-2 7B",
        "llama-2_7b_chat": "Llama-2 7B\nChat",
        "orca_7b": "Orca 7B",
        "phi-2": "Phi-2",
        "vicuna-13b-v1.5": "Vicuna 13B\nv1.5",
        "orca_13b": "Orca 13B",
        "llama-2_13b": "Llama-2 13B",
        "llama-2_13b_chat": "Llama-2 13B\nChat",
        "qwen_14b": "Qwen 14B",
        "qwen_14b_chat": "Qwen 14B\nChat",
        "cohere__command": "Cohere",
        "gpt-3.5-turbo": "OpenAI GPT 3.5",
        "gpt-4": "OpenAI GPT 4"
    })
    mode_map = dict({
        "Region": region_map,
        "Income group": income_map,
        "error_type": error_type_map,
        "Query": query_map,
        "llm": llm_map
    })

    if mode in mode_map:
        return mode_map[mode][s]
    else:
        raise ValueError(f"Unrecognized mode {mode}")
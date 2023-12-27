import pickle
import torch
import numpy as np
import os
from tqdm import tqdm
import json


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
    # let's strip away some things
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
            if words[-2] == 'in':
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

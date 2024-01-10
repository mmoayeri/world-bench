from abc import ABC, abstractmethod
from llm import LLM, _LLM_DICT
from typing import List
from my_utils import load_cached_data, cache_data
from constants import _CACHED_DATA_ROOT, _MOST_RECENT_YEAR
import os
import numpy as np
import pandas as pd

### Some GT computation helpers
# TODO: replace 2022 with some constant _MOST_RECENT_YEAR
def most_recent_val(row, starting_year=_MOST_RECENT_YEAR):
    for year in range(starting_year, starting_year-5, -1):
        if not np.isnan(row[str(year)]):
            break
    return row[str(year)]#, year

class Query(ABC):
    def __init__(self, question_template: str, nickname: str, trial: int=1):
        self.nickname = nickname
        self.question_template = question_template
        self.trial = self.trial
        # self.answer_root = os.path.join(_CACHED_DATA_ROOT, f'answers_trial{trial}')

    @abstractmethod
    def warm_up_llm(self, llm: LLM):
        pass

    def answer_path(self, llm_name: str):
        # return os.path.join(_CACHED_DATA_ROOT, 'answers', self.nickname, llm_name)
        # return os.path.join(_CACHED_DATA_ROOT, 'answers_trial2', self.nickname, llm_name)

        ext = self.gt_mode if self.gt_mode != 'avg' else ''
        return os.path.join(_CACHED_DATA_ROOT, f'answers_trial{self.trial}{ext}', self.nickname, llm_name)
        # return os.path.join(_CACHED_DATA_ROOT, 'answers', self.nickname, llm_name)

    def query(self, llm: LLM):
        """
        Each query will contain some general template question, with a space for a 'country' to fill in.
        Note that sometimes 'country' can be something else (e.g. Liga MX vs La Liga)
        """
        answer_path = self.answer_path(llm.get_modelname())
        if os.path.exists(answer_path + '.json'):
            answers_dict = load_cached_data(answer_path, mode='json')
            assert answers_dict['question_template'] == self.question_template, \
                f"Mismatch in question_template. Cached answers are for {answers_dict['question_template']}, while your query " \
                f"currently asks {self.question_template}. Change query.nickname or delete cached answers to avoid cache collision."
            answers_by_country = answers_dict['answers']
        else:
            questions = [self.question_template.format(country=country) for country in self.countries]
            answers = llm.answer_questions(questions, self.base_instruction, self.example)
            answers_by_country = dict({country:answer for country, answer in zip(self.countries, answers)})
            answers_dict = dict({
                'question_template': self.question_template,
                'answers': answers_by_country
            })
            for c, ans in answers_by_country.items():
                print(c, ans)
            print('CACHING TO ', answer_path)
            cache_data(answer_path, answers_dict, mode='json')
        return answers_by_country


class WorldBankQuery(Query, ABC):
    def __init__(self, metric: str, nickname: str, gt_csv_path: str, trial: int):
        self.gt_csv_path = gt_csv_path
        self.gt_mode = 'avg'
        self.trial = trial
        self.metric = metric
        self.nickname = nickname
        # question_template = f"What is the {self.metric} for the country " + "{country}? Be as concise as possible. Answer in no more than one sentence!"
        self.set_question_template()
        self.base_instruction = \
            f"I will ask you factual questions about countries. Specifically, I will ask you for the {self.metric}. " \
            f"You will answer as concisely as possible - only answer with the number! First I will give an example with the answer. " \
             "Then I will ask you my question, and you will provide the answer in the same way."

        # self.example = (question_template.format(country=eg_country), eg_answer)
        # eg_country, eg_answer = self.load_gt_answers()
        self.load_gt_answers()

        super().__init__(question_template=self.question_template, nickname=nickname)

    def set_gt_mode(self, gt_mode: str):
        if gt_mode != self.gt_mode:
            self.gt_mode = gt_mode
            self.load_gt_answers()
            self.set_question_template()

    def set_question_template(self):
        if 'specific_year' in self.gt_mode:
            self.question_template = f"What was the {self.metric} for the country " + "{country} in "
            self.question_template += f"{self.specific_year}? Do not answer in a complete sentence - only provide the number!"
        else:
            self.question_template = f"What is the {self.metric} for the country " + "{country}?  Do not answer in a complete sentence - only provide the number!"


    def load_gt_answers(self):
        cats = pd.read_csv('gt_answers/country_categories/region_and_income.csv')
        to_skip = [c.replace('\\', '').replace('\&', '&') for c in cats.iloc[220:].Economy] #+ ['Switzerland']

        df = pd.read_csv(self.gt_csv_path, skiprows=[0,1,2,3]).set_index('Country Name')
        df = df[~df.index.isin(to_skip)]

        if self.gt_mode == 'avg':
            df['gt_answer'] = df[['2020', '2021', '2022']].mean(1).round(3)
        elif 'most_recent' in self.gt_mode:
            gt_mode_split = self.gt_mode.split('_')
            if len(gt_mode_split) == 2:
                df['gt_answer'] = df.apply(most_recent_val, axis=1)
            else:
                self.starting_year = int(gt_mode_split[-1])
                df['gt_answer'] = df.apply(lambda x: most_recent_val(x, starting_year), axis=1)
        elif 'specific_year' in self.gt_mode:
            self.specific_year = int(self.gt_mode.split('_')[-1])
            df['gt_answer'] = df[str(self.specific_year)]
        else:
            raise ValueError(f"gt_mode {self.gt_mode} not recognized. Must be 'avg', 'most_recent', or in the form 'specific_year_2020'.")
        
        df = df.dropna(subset=['gt_answer']).reset_index()
        self.gt_df = df[['Country Name', 'gt_answer']]
        gt_df = df[['Country Name', 'gt_answer']]
        self.countries = list(self.gt_df['Country Name'])
        self.countries.remove('Switzerland')
        eg_country = 'Switzerland'
        eg_answer = self.gt_df[self.gt_df['Country Name'] == 'Switzerland']['gt_answer'].iloc[0]
        if np.isnan(eg_answer):
            # fall back to average of recent years -- this is just an example value anyways
            eg_answer = df.loc['Switzerland'][['2020','2021','2022']].mean()
        
        if eg_answer == int(eg_answer):
            eg_answer = format(int(eg_answer), ",")

        self.example = (self.question_template.format(country=eg_country), eg_answer)
        # return eg_country, str(eg_answer)

    # def init_base_question(self):
    #     self.base_question = [
    #         {'role': 'user', 'content': f'What is the {self.metric} for the country {eg_country}? Be as concise as possible. Answer in no more than one sentence!'},
    #         {'role': 'assistant', 'content': f'The {self.metric} for the country {eg_country} is {eg_answer}'}
    #         # {'role': 'user', 'content': 'What is the maternal mortality ratio as number of deaths per 100,000 live births for the country Egypt? Be as concise as possible. Answer in no more than one sentence!'}
    #         ]


    def warm_up_llm(self, llm):
        # ind = np.random.choice(len(self.gt_df))
        # eg_country, eg_answer = [self.gt_df.iloc[ind][x] for x in ['Country Name', 'gt_answer']]
        eg_country = 'Switzerland'
        eg_answer = self.gt_df[self.gt_df['Country Name'] == 'Switzerland']['gt_answer'].iloc[0]

        question = self.question_template.format(country=eg_country)
        msg = f"I will ask you factual questions about countries. Specifically, I will ask you for the {self.metric}. " \
              f"You will answer as concisely as possible. For example, I will ask '{question}' and you will answer '{eg_answer}'"
        answer = llm.answer_questions([msg])
        print(msg, answer)
        answer = llm.answer_questions([question])
        print(answer, eg_answer, eg_country, question)
        

class EducationExpenditureQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="government expenditure on education as a total percent of GDP", 
            nickname="education_expenditure",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/education_expenditure.csv",
            trial=trial
        )

class PopulationQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="total population", 
            nickname="population",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/population.csv",
            trial=trial
        )

class WomenInParliamentQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="proportion of seats held by women in national parliaments (as a percent)", 
            nickname="women_in_parliament",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/parliament_seats_gender_ratio.csv",
            trial=trial
        )

class UnemploymentQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="unemployment as a percent of the total labor force", 
            nickname="unemployment",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/unemployment.csv",
            trial=trial
        )

class MaternalMortalityQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="maternal mortality ratio as number of deaths per 100,000 live births", 
            nickname="maternal_mortality_ratio",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/maternal_mortality_ratio.csv",
            trial=trial
        )

class ElectricityAccessQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="percent of the total population that has access to electricity", 
            nickname="electricity_access_percent",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/access_to_electricity_percent.csv",
            trial=trial
        )

class AgriculturalLandQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="percent of total land area that is agricultural", 
            nickname="agricultural_land_percent",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/agricultural_land_percent.csv",
            trial=trial
        )

class CO2EmissionsQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="amount of carbon dioxide emissions in metric tonnes per capita", 
            nickname="co2_emissions",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/co2_emissions_metric_tonnes_per_capita.csv",
            trial=trial
        )

class GDPQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="GDP measured in US dollars", 
            nickname="gdp",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/gdp_current_usd.csv",
            trial=trial
        )

class GDPPerPersonQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="GDP at purchasing power parity (PPP) per person employed", 
            nickname="gdp_ppp",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/gdp_per_person_employed.csv",
            trial=trial
        )

class RenewableEnergyQuery(WorldBankQuery):
    def __init__(self, trial: int=1):
        super().__init__(
            metric="renewable energy consumption as a percent of total final energy consumption", 
            nickname="renewable_percent",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/renewable_energy_consumption.csv",
            trial=trial
        )

_QUERY_DICT = dict({
    "population": PopulationQuery, 
    "unemployment": UnemploymentQuery,
    "maternal_mortality_ratio": MaternalMortalityQuery,
    "women_in_parliament": WomenInParliamentQuery,
    "education_expenditure": EducationExpenditureQuery,
    "electricity_access_percent": ElectricityAccessQuery,
    "agricultural_land_percent": AgriculturalLandQuery,
    "co2_emissions": CO2EmissionsQuery,
    "gdp": GDPQuery,
    "gdp_ppp": GDPPerPersonQuery,
    "renewable_percent": RenewableEnergyQuery,
})

def run_all_queries_for_llm(llm_info, trial=1, gt_mode='avg'):
    make_llm, llm_key = llm_info
    llm = make_llm(llm_key)
    for _, init_query in _QUERY_DICT.items():
        query = init_query(trial)
        query.set_gt_mode(gt_mode)
        print(f"Processing query {query.nickname} for LLM {llm.get_modelname()}")
        query.query(llm)
    # del llm.model; torch.cuda.empty_cache()



if __name__ == '__main__':
    # query = PopulationQuery()

    # llm = Vicuna()

    import submitit
    executor = submitit.AutoExecutor(folder='./logs/')
    executor.update_parameters(
        timeout_min=300,  
        slurm_partition="tron", 
        slurm_qos="default", 
        slurm_account="nexus", 
        mem_gb=32, 
        tasks_per_node=1,
        slurm_gres='gpu:rtxa6000'
    )
    jobs = []
    trial = 5
    with executor.batch():
        for llm_name, llm_info in _LLM_DICT.items():
            # if 'qwen' in llm_name or 'vicuna-13b-v1.5' in llm_name or 'llama-2_13b' in llm_name:
            jobs.append(executor.submit(run_all_queries_for_llm, llm_info, trial))
    outputs = [job.result() for job in jobs]
    print(outputs)


    # llm_name = 'gpt-3.5-turbo'
    # llm_name = 'cohere__command'
    # llm_name = 'gpt-4'
    # make_llm, llm_key = _LLM_DICT[llm_name]
    # llm = make_llm(llm_key)

    # import torch
    # for _, (make_llm, llm_key) in _LLM_DICT.items():
    #     llm = make_llm(llm_key)
    # for _, init_query in _QUERY_DICT.items():
    #     query = init_query()
    #     print(f"Processing query {query.nickname} for LLM {llm.get_modelname()}")
    #     query.query(llm)

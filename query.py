from abc import ABC, abstractmethod
from llm import LLM, Vicuna
from typing import List
from my_utils import load_cached_data, cache_data
from constants import _CACHED_DATA_ROOT
import os
import numpy as np
import pandas as pd

class Query(ABC):
    def __init__(self, question_template: str, nickname: str):
        self.nickname = nickname
        self.question_template = question_template
        self.answer_root = os.path.join(_CACHED_DATA_ROOT, 'answers')

    @abstractmethod
    def warm_up_llm(self, llm: LLM):
        pass

    def answer_path(self, llm_name: str):
        return os.path.join(_CACHED_DATA_ROOT, 'answers', self.nickname, llm_name)

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
            # self.warm_up_llm(llm)
            # answers = llm.answer_questions([self.question_template.format(country=country) for country in self.countries])
            answers = llm.answer_questions([
                    [*self.base_question, 
                    {'role': 'user', 'content': self.question_template.format(country=country)}]
                for country in self.countries])
            answers_by_country = dict({country:answer for country, answer in zip(self.countries, answers)})
            answers_dict = dict({
                'question_template': self.question_template,
                'answers': answers_by_country
            })
            for c, ans in answers_by_country.items():
                print(c, ans)
            cache_data(answer_path, answers_dict, mode='json')
        return answers_by_country

    # def eval(self, gt_answers: Dict[str, str], metric_fn):

class WorldBankQuery(Query, ABC):
    def __init__(self, metric: str, nickname: str, gt_csv_path: str):
        self.gt_csv_path = gt_csv_path
        self.load_gt_answers()
        self.metric = metric
        self.init_base_question()

        question_template = f"What is the {self.metric} for the country " + "{country}? Be as concise as possible. Answer in no more than one sentence!"
        # question_template = f"What is the {metric} for the country " + "{country}?  Do not answer in a complete sentence - only provide the number!"
        super().__init__(question_template=question_template, nickname=nickname)

    def load_gt_answers(self):
        df = pd.read_csv(self.gt_csv_path, skiprows=[0,1,2,3])
        df['gt_answer'] = np.nanmean([df['2022'], df['2021'], df['2020']], axis=0)
        gt_df = df[['Country Name', 'gt_answer']]
        self.gt_df = gt_df[~np.isnan(gt_df.gt_answer)]
        self.countries = list(self.gt_df['Country Name'])

    def init_base_question(self):
        eg_country = 'Switzerland'
        eg_answer = self.gt_df[self.gt_df['Country Name'] == 'Switzerland']['gt_answer'].iloc[0]
        self.base_question = [
            {'role': 'user', 'content': f'What is the {self.metric} for the country {eg_country}? Be as concise as possible. Answer in no more than one sentence!'},
            {'role': 'assistant', 'content': f'The {self.metric} for the country {eg_country} is {eg_answer}'}
            # {'role': 'user', 'content': 'What is the maternal mortality ratio as number of deaths per 100,000 live births for the country Egypt? Be as concise as possible. Answer in no more than one sentence!'}
            ]


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
    def __init__(self):
        super().__init__(
            metric="government expenditure on education as a total percent of GDP", 
            nickname="education_expenditure",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/education_expenditure.csv"
        )

class PopulationQuery(WorldBankQuery):
    def __init__(self):
        super().__init__(
            metric="total population", 
            nickname="population",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/population.csv"
        )

class WomenInParliamentQuery(WorldBankQuery):
    def __init__(self):
        super().__init__(
            metric="proportion of seats held by women in national parliaments (as a percent)", 
            nickname="women_in_parliament",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/parliament_seats_gender_ratio.csv"
        )

class UnemploymentQuery(WorldBankQuery):
    def __init__(self):
        super().__init__(
            metric="unemployment as a percent of the total labor force", 
            nickname="unemployment",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/unemployment.csv"
        )

class MaternalMortalityQuery(WorldBankQuery):
    def __init__(self):
        super().__init__(
            metric="maternal mortality ratio as number of deaths per 100,000 live births", 
            nickname="maternal_mortality_ratio",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/maternal_mortality_ratio.csv"
        )

class ElectricityAccessQuery(WorldBankQuery):
    def __init__(self):
        super().__init__(
            metric="percent of the total population that has access to electricity", 
            nickname="electricity_access_percent",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/access_to_electricity_percent.csv"
        )

class AgriculturalLandQuery(WorldBankQuery):
    def __init__(self):
        super().__init__(
            metric="percent of total land area that is agricultural", 
            nickname="agricultural_land_percent",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/agricultural_land_percent.csv"
        )

class CO2EmissionsQuery(WorldBankQuery):
    def __init__(self):
        super().__init__(
            metric="amount of carbon dioxide emissions in metric tonnes per capita", 
            nickname="co2_emissions",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/co2_emissions_metric_tonnes_per_capita.csv"
        )

class GDPQuery(WorldBankQuery):
    def __init__(self):
        super().__init__(
            metric="GDP measured in US dollars", 
            nickname="gdp",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/gdp_current_usd.csv"
        )

class GDPPerPersonQuery(WorldBankQuery):
    def __init__(self):
        super().__init__(
            metric="GDP at purchasing power parity (PPP) per person employed", 
            nickname="gdp_ppp",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/gdp_per_person_employed.csv"
        )

class RenewableEnergyQuery(WorldBankQuery):
    def __init__(self):
        super().__init__(
            metric="renewable energy consumption as a percent of total final energy consumption", 
            nickname="renewable_percent",
            gt_csv_path="/cmlscratch/mmoayeri/world-qa/gt_answers/renewable_energy_consumption.csv"
        )

_QUERY_DICT = dict({
    # "population": PopulationQuery, 
    # "unemployment": UnemploymentQuery,
    # "maternal_mortality_ratio": MaternalMortalityQuery,
    # "women_in_parliament": WomenInParliamentQuery,
    # "education_expenditure": EducationExpenditureQuery,
    "electricity_access_percent": ElectricityAccessQuery,
    "agricultural_land_percent": AgriculturalLandQuery,
    "co2_emissions": CO2EmissionsQuery,
    "gdp": GDPQuery,
    "gdp_ppp": GDPPerPersonQuery,
    "renewable_percent": RenewableEnergyQuery,
})


if __name__ == '__main__':
    # query = PopulationQuery()

    llm = Vicuna()
    for _, init_query in _QUERY_DICT.items():
        query = init_query()
        query.query(llm)
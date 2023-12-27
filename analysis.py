import pandas as pd
import numpy as np
from query import _QUERY_DICT
from llm import Vicuna, _LLM_DICT
import numpy as np
from typing import List, Tuple
from my_utils import parse_number_str, load_cached_data, cache_data
from abc import ABC
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_theme(style="darkgrid")
plt.rcParams['text.usetex'] = True

# plt.rcParams.update({
#     "font.family": "sans-serif",
#     "font.sans-serif": "cmss10",
#     "axes.formatter.use_mathtext": "True"
# })

"""
Ok, let's put together a good hour of work. 

We want to clean up this analysis a bit. We now have country categorization annotations too,
so we want to be able to return a master dataframe with all info we are interested in...

which is...
1. A df that has GT and LLM answers for a list of queries
2. A df that applies some error metric to the above df, so each cell provides the error for a given 

Ok, I have this now. I still need to incorporate the country categorizations. I checked error correlations
and get nothing lol. But country categorizations are more informative. 

Let's be straightforward. Plan is: 
1. load country categories into errors_df
2. Plot error vs. category for each query x each error type
"""

class ErrorMetric(ABC):
    def compute_errors_df(self, answers_df: pd.DataFrame, gts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects answers_df and gts_df to have identical index and column names, as outputted by
        the collect_responses_and_gts function.
        """
        return answers_df.combine(gts_df, self.error)

    def error(self, ans, gt):
        raise NotImplementedError

class RelError(ErrorMetric):
    def error(self, ans, gt):
        return (ans - gt) / np.maximum(ans, gt)

class AbsRelError(ErrorMetric):
    def error(self, ans, gt):
        return np.abs(ans - gt) / np.maximum(ans, gt)

class AbsError(ErrorMetric):
    def error(self, ans, gt):
        return np.abs(ans - gt)

class RawError(ErrorMetric):
    def error(self, ans, gt):
        return ans - gt

# class Analyzer():
#     def collect_responses_and_gts(self, query_names: List[str] = [], llm_name: str = 'vicuna') -> pd.Dataframe:
def collect_responses_and_gts(query_names: List[str] = [], llm_name: str = 'vicuna-13b-v1.5') -> Tuple[pd.DataFrame, pd.DataFrame]:
    # if no query_names are provided, return all responses and GTs to all queries
    if query_names == []:
        query_names = list(_QUERY_DICT.keys())

    answers, gts = dict(), dict()
    for query_name in query_names:
        # Some queries have gts/answers for diff country sets than others...
        # So let's just collect them all first
        query = _QUERY_DICT[query_name]()
        llm = _LLM_DICT[llm_name]
        curr_answers = query.query(llm)
        answers[query.nickname] = dict({c:parse_number_str(v) for c,v in curr_answers.items()})
        gt_df = query.gt_df.set_index('Country Name')
        gts[query.nickname] = dict({c:gt_df.loc[c]['gt_answer'] for c in answers[query.nickname]})
    
    # Now let's get the set of countries
    all_countries = []
    for _, query_answers in answers.items():
        for c in query_answers:
            if c not in all_countries:
                all_countries.append(c)
    
    answers_df_dict, gts_df_dict = dict(), dict()
    for c in all_countries:
        answers_df_dict[c] = [answers[query_name][c] if c in answers[query_name] else np.nan for query_name in query_names]
        gts_df_dict[c] = [gts[query_name][c] if c in gts[query_name] else np.nan for query_name in query_names]
    
    answers_df, gts_df = [pd.DataFrame.from_dict(df_dict, orient='index', columns=query_names)
                            for df_dict in [answers_df_dict, gts_df_dict]]

    return answers_df, gts_df

def check_error_correlation():
    # this failed :/ no error correlation
    answers_df, gts_df = collect_responses_and_gts()
    for metric in [RawError, AbsError, AbsRelError, RelError]:
        errors_df = metric().compute_errors_df(answers_df2, gts_df2)
        columns = errors_df.columns
        for i, c1 in tqdm(enumerate(columns)):
            for c2 in columns[:i]:
                sub_df = errors_df[[c1, c2]]
                sub_df = sub_df.dropna()
                correlations[(c1, c2)] = pearsonr(sub_df[c1], sub_df[c2])
        print(correlations)
        print()

### What we really care about: performance by country / income level
def error_by_category():
    categories_df = pd.read_csv('gt_answers/country_categories/region_and_income.csv').set_index('Economy')
    answers_df, gts_df = collect_responses_and_gts()

    error_metrics = [AbsRelError, RelError, RawError, AbsError]#, 

    f, axs = plt.subplots(len(error_metrics), 2, figsize=(10, 5*len(error_metrics)))
    for i, error_metric in enumerate(error_metrics):
        metric_name = error_metric.__name__

        errors_df = error_metric().compute_errors_df(answers_df, gts_df)

        if (metric_name == 'RawError') or (metric_name == 'AbsError'):
            errors_df = errors_df.drop('population', axis=1)
            
        errors_df = errors_df[errors_df.index.isin(categories_df.index)]
        errors_df = errors_df.dropna()

        errors_df['region'] = categories_df.Region.loc[errors_df.index]

        grouped = errors_df.groupby('region').mean()
        df_melted = pd.melt(grouped.reset_index(), id_vars='region', var_name='column', value_name='value')
        sns.barplot(data=df_melted, x="column", y="value", hue="region", palette=sns.color_palette('bright'), ax=axs[i][0],
            hue_order=["Sub-Saharan Africa", "East Asia \& Pacific", "Middle East \& North Africa",  "South Asia", "Latin America \& Caribbean","Europe \& Central Asia", "North America"])
        axs[i][0].set_xticklabels(axs[i][0].get_xticklabels(), rotation=45)
        axs[i][0].set_ylabel(metric_name)

        errors_df = errors_df.drop('region', axis=1)
        errors_df['income'] = categories_df['Income group'].loc[errors_df.index]
        grouped = errors_df.groupby('income').mean()
        df_melted = pd.melt(grouped.reset_index(), id_vars='income', var_name='column', value_name='value')
        sns.barplot(data=df_melted, x="column", y="value", hue="income", palette=sns.color_palette('bright'), ax=axs[i][1],
                hue_order=["Low income", "Lower middle income", "Upper middle income", "High income"])
        axs[i][1].set_ylabel(metric_name)
        axs[i][1].set_xticklabels(axs[i][1].get_xticklabels(), rotation=45)
    
    f.tight_layout(); f.savefig('plots/errors_by_region_and_income.jpg', dpi=300)

    # TODO: add average plots as well, for AbsRelError

    f, ax = plt.subplots(1,1);
    sns.barplot(data=df_melted, x="region", y="value", ax =ax, order=["Sub-Saharan Africa", "East Asia \& Pacific", "Middle East \&North Africa",  "South Asia", "Latin America \& Caribbean","Europe \& Central Asia", "North America"], ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel('Average Absolute Relative Error')
    f.tight_layout(); f.savefig('plots/avg_absrelerror_region.jpg', dpi=200)

if __name__ == '__main__':
    error_by_category()
            

    

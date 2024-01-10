import pandas as pd
import numpy as np
from query import _QUERY_DICT
from llm import LLM, _LLM_DICT
import numpy as np
from typing import List, Tuple
from my_utils import parse, load_cached_data, cache_data, beautify
from abc import ABC
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

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

def to_num(parsed_answer: str):
    try:
        num = float(parsed_answer)
    except:
        num = np.nan
    return num

# def collect_responses_and_gts(query_names: List[str] = [], llm_name: str = 'vicuna-13b-v1.5') -> Tuple[pd.DataFrame, pd.DataFrame]:
def collect_responses_and_gts(
    query_names: List[str] = [], 
    llm_name: str = 'vicuna-13b-v1.5', 
    trial: int=1,
    gt_mode: str = 'most_recent'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # if no query_names are provided, return all responses and GTs to all queries
    if query_names == []:
        query_names = list(_QUERY_DICT.keys())

    cats = pd.read_csv('gt_answers/country_categories/region_and_income.csv')
    to_skip = [c.replace('\\', '').replace('\&', '&') for c in cats.iloc[220:].Economy] + ['Switzerland']

    answers, gts = dict(), dict()
    for query_name in query_names:
        # Some queries have gts/answers for diff country sets than others...
        # So let's just collect them all first
        query = _QUERY_DICT[query_name](trial=trial)
        query.change_gt_mode(gt_mode)
        make_llm, llm_key = _LLM_DICT[llm_name]
        llm = make_llm(llm_key)

        curr_answers = query.query(llm)
        gt_df = query.gt_df.set_index('Country Name')
        countries_w_gt_and_ans = [c for c in curr_answers if ((c in gt_df.index) and (c not in to_skip))]
        answers[query.nickname] = dict({c:to_num(parse(curr_answers[c])) for c in countries_w_gt_and_ans})
        gts[query.nickname] = dict({c:gt_df.loc[c]['gt_answer'] for c in countries_w_gt_and_ans})
        # answers[query.nickname] = dict({c:to_num(parse(v)) for c,v in curr_answers.items() if c not in to_skip})
        # gts[query.nickname] = dict({c:gt_df.loc[c]['gt_answer'] for c in answers[query.nickname] if c not in to_skip})
    
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
    # It failed bc I implemented it wrong. I should compare gts_df of a col to the errors
    answers_df, gts_df = collect_responses_and_gts()
    for metric in [AbsRelError, RelError]:
        errors_df = metric().compute_errors_df(answers_df, gts_df)
        columns = errors_df.columns
        for i, c1 in tqdm(enumerate(columns)):
            for c2 in columns[:i]:
                # this is wrong
                sub_df = errors_df[[c1, c2]]
                sub_df = sub_df.dropna()
                correlations[(c1, c2)] = pearsonr(sub_df[c1], sub_df[c2])
        print(correlations)
        print()

### What we really care about: performance by country / income level
def error_by_category(llm_name: str):
    categories_df = pd.read_csv('gt_answers/country_categories/region_and_income.csv').set_index('Economy')
    answers_df, gts_df = collect_responses_and_gts(llm_name = llm_name)

    error_metrics = [AbsRelError, RelError]#, RawError, AbsError]#, 

    plot_root = f"plots_2021/{llm_name}/"
    os.makedirs(plot_root, exist_ok=True)
    f, axs = plt.subplots(len(error_metrics), 2, figsize=(15, 5*len(error_metrics)))
    f2, axs2 = plt.subplots(len(error_metrics), 2, figsize=(8, 4*len(error_metrics)));
    for i, error_metric in enumerate(error_metrics):
        metric_name = error_metric.__name__

        errors_df = error_metric().compute_errors_df(answers_df, gts_df)

        if (metric_name == 'RawError') or (metric_name == 'AbsError'):
            errors_df = errors_df.drop('population', axis=1)
            
        errors_df = errors_df[errors_df.index.isin(categories_df.index)]
        # errors_df = errors_df.dropna()

        errors_df['region'] = categories_df.Region.loc[errors_df.index]

        grouped = errors_df.groupby('region').median()
        df_melted = pd.melt(grouped.reset_index(), id_vars='region', var_name='column', value_name='value')
        sns.barplot(data=df_melted, x="column", y="value", hue="region", palette=sns.color_palette('bright'), ax=axs[i][0],
            hue_order=["Sub-Saharan Africa", "East Asia \& Pacific", "Middle East \& North Africa",  "South Asia", "Latin America \& Caribbean","Europe \& Central Asia", "North America"])
        axs[i][0].legend(ncol=2)
        axs[i][0].set_xticklabels(axs[i][0].get_xticklabels(), rotation=45)
        axs[i][0].set_ylabel(metric_name.title())

        sns.barplot(data=df_melted, x="region", y="value", ax=axs2[i][0], order=["Sub-Saharan Africa", "East Asia \& Pacific", "Middle East \& North Africa",  "South Asia", "Latin America \& Caribbean","Europe \& Central Asia", "North America"], ci=None)
        axs2[i][0].set_xticklabels(axs2[i][0].get_xticklabels(), rotation=90)
        axs2[i][0].set_ylabel(metric_name.title())

        errors_df = errors_df.drop('region', axis=1)
        errors_df['income'] = categories_df['Income group'].loc[errors_df.index]
        grouped = errors_df.groupby('income').median()
        df_melted = pd.melt(grouped.reset_index(), id_vars='income', var_name='column', value_name='value')
        sns.barplot(data=df_melted, x="column", y="value", hue="income", palette=sns.color_palette('bright'), ax=axs[i][1],
                hue_order=["Low income", "Lower middle income", "Upper middle income", "High income"])
        axs[i][1].set_ylabel(metric_name)
        axs[i][1].set_xticklabels(axs[i][1].get_xticklabels(), rotation=45)
    
        sns.barplot(data=df_melted, x="income", y="value", ax=axs2[i][1], order=["Low income", "Lower middle income", "Upper middle income", "High income"], ci=None)
        axs2[i][1].set_xticklabels(axs2[i][1].get_xticklabels(), rotation=90)
        axs2[i][1].set_ylabel(metric_name.title())
    
    f.tight_layout(); f.savefig(f"{plot_root}/full.jpg", dpi=300, bbox_inches='tight')
    f2.tight_layout(); f2.savefig(f'{plot_root}/avg.jpg', dpi=300, bbox_inches='tight')

def plot_all_llm_errors():
    for llm_name in tqdm(_LLM_DICT):
        try:
            error_by_category(llm_name)
        except Exception as e:
            print(f"Failed for {llm_name} with error {e}")

def consolidate_results(trial: int = 1):
    """
    Desired columns: error type, llm, query, country, region, income level
    
    How to populate: 
    1. loop through error type and llms (collect_responses_and_gts loops through queries)
    2. add annotations of error type and llm as columns w/ same val
    """
    categories_df = pd.read_csv('gt_answers/country_categories/region_and_income.csv').set_index('Economy')
    all_dfs = []
    for llm_name in _LLM_DICT:
        answers_df, gts_df = collect_responses_and_gts(llm_name = llm_name, trial=trial)
        for error_metric in [AbsRelError, RelError]:
            errors_df = error_metric().compute_errors_df(answers_df, gts_df)
            errors_df = errors_df[errors_df.index.isin(categories_df.index)]
            df = pd.melt(errors_df.reset_index(), id_vars='index', var_name ='Query')
            df = df.rename(columns={'index':'Economy', 'value': 'Error'})
            df['llm'] = [llm_name] * len(df)
            df['error_type'] = [error_metric.__name__] * len(df)
            df['Region'] = df.Economy.apply(lambda x : categories_df.Region.loc[x])
            df['Income group'] = df.Economy.apply(lambda x : categories_df['Income group'].loc[x])
            all_dfs.append(df)

    full_df = pd.concat(all_dfs)
    return full_df

def add_newline_everyother_space(s):
    return ''.join([w+'\n' if i %2 ==1 else w+' ' for i,w in enumerate(s.split(' '))])


def compute_disparities(full_df: pd.DataFrame, cat: str, error_type: str, id_var: str):
    sub_df = full_df[full_df.error_type == error_type]
    if id_var == 'both':
        grouped_df = sub_df.groupby(['llm', 'Query'])
    else:
        assert id_var in ['llm', 'Query'], f"id_var {id_var} is invalid. Must be either 'llm', 'Query', or 'both'"
        grouped_df = sub_df.groupby(id_var)

    delta = lambda df: df.groupby(cat).median().Error.max() - df.groupby(cat).median().Error.min()
    disparity_series = grouped_df.apply(delta)
    disparity_df = disparity_series.rename('Disparity').reset_index().sort_values('Disparity')
    return disparity_df

### Plotting

def by_category(full_df=None):
    if full_df is None:
        full_df = consolidate_results()
    
    # We exclude rel error when avging over queries, since the significance of the sign of the error is not 
    # consistent over queries: overestimating GDP is much different than overestimating Maternal Mortality Ratio
    df = full_df[full_df.error_type == 'AbsRelError']

    f, axs = plt.subplots(1, 2, figsize=(13,4))
    for ax, cat in zip(axs, ["Region", "Income group"]):
        avg = df.groupby(cat).median().sort_values('Error').reset_index()
        # avg = df.groupby(cat).median().sort_values('Error', ascending=Tru).reset_index()
        avg[cat] = avg[cat].apply(lambda s: beautify(s, mode=cat))
        sns.barplot(data=avg, x=cat, y="Error", ci=None, ax=ax)
        ax.bar_label(ax.containers[0], fmt="%0.3f", label_type='center', color='white')
        ax.set_ylabel("Absolute Relative Error")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8.5)
    f.tight_layout(); f.savefig('plots_2021/by_category.jpg', dpi=300, bbox_inches='tight')

def by_query(full_df=None):
    if full_df is None:
        full_df = consolidate_results()

    f, axs = plt.subplots(1, 2, figsize=(6,5))
    for ax, (error_type, df) in zip(axs, full_df.groupby("error_type")):
        avg = df.groupby("Query").median().sort_values("Error").reset_index()
        avg["Query"] = avg["Query"].apply(lambda x: beautify(x, mode="Query"))
        sns.barplot(data=avg, y="Query", x="Error", ci=None, ax=ax)
        # ax.bar_label(ax.containers[0], fmt="%0.3f", label_type='center', fontsize=8.5)
        ax.set_xlabel(beautify(error_type, mode='error_type'))
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8.5)
    f.tight_layout(); f.savefig('plots_2021/by_query.jpg', dpi=300, bbox_inches='tight')

def by_llm(full_df=None):
    if full_df is None:
        full_df = consolidate_results()

    f, axs = plt.subplots(1, 2, figsize=(6,5))
    for ax, (error_type, df) in zip(axs, full_df.groupby("error_type")):
        avg = df.groupby("llm").median().sort_values("Error").reset_index()
        avg["llm"] = avg["llm"].apply(lambda x: beautify(x, mode="llm"))

        sns.violinplot(data=df, y="llm", x="Error", ax = ax, inner=None)
        # sns.barplot(data=avg, y="llm", x="Error", ci=None, ax=ax)
        # ax.bar_label(ax.containers[0], fmt="%0.3f", label_type='center', fontsize=8.5)
        ax.set_xlabel(beautify(error_type, mode='error_type'))
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8.5)
        ax.set_ylabel("Language Model")
    f.tight_layout(); f.savefig('plots_2021/by_llm.jpg', dpi=300, bbox_inches='tight')

def by_category_and_query(full_df=None):
    if full_df is None:
        full_df = consolidate_results()

    orders = dict({
        "Region": ['North America', 'Europe \& Central Asia', 'Latin America \& Caribbean', 'South Asia', 
        'Middle East \& North Africa', 'East Asia \& Pacific', 'Sub-Saharan Africa'],
        "Income group" : ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    })

    f, axs = plt.subplots(2, 2, figsize=(15,10))
    for ax_row, (error_type, df) in zip(axs, full_df.groupby("error_type")):
        for ax, cat in zip(ax_row, ["Region", "Income group"]):
            avg = df.groupby(["Query", cat]).median().sort_values("Error").reset_index()
            avg["Query"] = avg["Query"].apply(lambda x: beautify(x, mode="Query"))
            avg[cat] = avg[cat].apply(lambda x: beautify(x, mode=cat).replace('\n', ' '))
            # order = list(avg.groupby(cat).median().sort_values('Error').index)
            order = [beautify(x, mode=cat).replace('\n',' ') for x in orders[cat]]
            sns.barplot(data=avg, x="Query", y="Error", hue=cat, hue_order=order, ci=None, ax=ax)
            # sns.barplot(data=avg, x="Query", y="Error", hue=cat, ci=None, ax=ax)
            ax.set_ylabel(beautify(error_type, mode='error_type'))
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8.5, rotation='vertical')
            ax.legend(fontsize=9)
    f.tight_layout(); f.savefig('plots_2021/by_cat_and_query.jpg', dpi=300, bbox_inches='tight')

def by_category_and_llm(full_df=None):
    if full_df is None:
        full_df = consolidate_results()

    orders = dict({
        "Region": ['North America', 'Europe \& Central Asia', 'Latin America \& Caribbean', 'South Asia', 
        'Middle East \& North Africa', 'East Asia \& Pacific', 'Sub-Saharan Africa'],
        "Income group" : ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    })

    f, axs = plt.subplots(2, 2, figsize=(15,10))
    for ax_row, (error_type, df) in zip(axs, full_df.groupby("error_type")):
        for ax, cat in zip(ax_row, ["Region", "Income group"]):
            avg = df.groupby(["llm", cat]).median().sort_values("Error").reset_index()
            avg["llm"] = avg["llm"].apply(lambda x: beautify(x, mode="llm"))
            avg[cat] = avg[cat].apply(lambda x: beautify(x, mode=cat).replace('\n', ' '))
            # order = list(avg.groupby(cat).median().sort_values('Error').index)
            order = [beautify(x, mode=cat).replace('\n',' ') for x in orders[cat]]
            sns.barplot(data=avg, x="llm", y="Error", hue=cat, hue_order=order, ci=None, ax=ax)
            # sns.barplot(data=avg, x="Query", y="Error", hue=cat, ci=None, ax=ax)
            ax.set_ylabel(beautify(error_type, mode='error_type'))
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8.5, rotation='vertical')
            ax.legend(fontsize=9)
    f.tight_layout(); f.savefig('plots_2021/by_cat_and_llm.jpg', dpi=300, bbox_inches='tight')
    
def by_country(full_df=None, n_in_tail=20):
    if full_df is None:
        full_df = consolidate_results()

    # We exclude rel error when avging over queries, since the significance of the sign of the error is not 
    # consistent over queries: overestimating GDP is much different than overestimating Maternal Mortality Ratio
    df = full_df[full_df.error_type == 'AbsRelError']

    f, ax = plt.subplots(1, 1, figsize=(13,4))
    avg = df.groupby('Economy').median().sort_values('Error').reset_index()
    avg = pd.concat([avg.iloc[:n_in_tail], avg.iloc[-1*n_in_tail:]])
    sns.barplot(data=avg, x="Economy", y="Error", ci=None, ax=ax)
    ax.set_ylabel('Absolute Relative Error')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation='vertical')
    f.tight_layout(); f.savefig('plots_2021/by_country.jpg', dpi=300, bbox_inches='tight')


def disparity(full_df: pd.DataFrame=None, id_var: str='llm', error_type='AbsRelError'):
    if full_df is None:
        full_df = consolidate_results()

    f, axs = plt.subplots(1, 2, figsize=(7,5))
    for ax, cat in zip(axs, ["Region", "Income group"]):
        df = compute_disparities(full_df, cat, error_type, id_var)
        df[id_var] = df[id_var].apply(lambda x: beautify(x, mode=id_var))
        sns.barplot(data=df, y=id_var, x="Disparity", ci=None, ax=ax)
        ax.bar_label(ax.containers[0], fmt="%0.3f", label_type='center', fontsize=8.5)
        ax.set_xlabel(f"{beautify(error_type, mode='error_type')}\nDisparity over {cat}s", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8.5)
    f.tight_layout(); f.savefig(f'plots_2021/by_{id_var}_disparity_{error_type}.jpg', dpi=300, bbox_inches='tight')

def error_vs_disparity(id_var='llm', cat="Region"):
    from scipy.stats import pearsonr
    full_df = consolidate_results()
    disparity_df = compute_disparities(full_df, cat, 'AbsRelError', id_var)
    df = full_df[full_df.error_type == 'AbsRelError']

    sub_df = df.groupby(id_var).median()
    disparity_df['error'] = disparity_df[id_var].apply(lambda x: sub_df.loc[x])
    print(pearsonr(disparity_df.Disparity, disparity_df.error))



if __name__ == '__main__':
    # error_by_category()
    # plot_all_llm_errors()

    full_df = consolidate_results()

    by_category(full_df)
    by_query(full_df)
    by_llm(full_df)
    # by_country(full_df)
    by_category_and_query(full_df)
    by_category_and_llm(full_df)

    for id_var in ['llm', 'Query']:
        disparity(full_df, id_var)
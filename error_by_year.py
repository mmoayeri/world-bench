from analysis import *
from llm import _LLM_DICT

error_metric = AbsRelError()
all_errs = dict()
years = list(range(2012, 2023))
f, axs = plt.subplots(3,6, figsize=(18,12))
_LLM_DICT = _LLM_DICT
for ax, llm_name in tqdm(zip(axs.ravel(), _LLM_DICT), total=min(len(_LLM_DICT), len(axs.ravel()))):
    errs = []
    for year in years:
        answers_df, gts_df = collect_responses_and_gts(llm_name = llm_name, gt_mode=f'specific_year_{year}')
        errs.append(error_metric.compute_errors_df(answers_df, gts_df).stack().mean())
    all_errs[llm_name] = errs
    ax.plot(years, errs, label='Specific year (x-axis)', color='blue')
    answers_df, gts_df = collect_responses_and_gts(llm_name = llm_name)
    ax.axhline(y=error_metric.compute_errors_df(answers_df, gts_df).stack().mean(), 
        label="Avg over '20, '21, '22", ls='--', color='coral')

    answers_df, gts_df = collect_responses_and_gts(llm_name = llm_name, gt_mode=f'most_recent')
    ax.axhline(y=error_metric.compute_errors_df(answers_df, gts_df).stack().mean(), 
        label="Most Recent Year", ls='--', color='deepskyblue')
    ax.set_title(beautify(llm_name, mode='llm').replace('\n',' '))
    ax.legend(title='GT Selection')

f.tight_layout(); f.savefig('plots/error_by_year.jpg', dpi=300, bbox_inches='tight')
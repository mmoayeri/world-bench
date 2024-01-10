from scipy.stats import pearsonr


for llm_name in _LLM_DICT:
    all_answers =[]
    for trial in range(1,6):
        answers_df, gts_df = collect_responses_and_gts(llm_name = llm_name, trial=trial)
        all_answers.append(answers_df)

    all_answers = pd.concat(all_answers)
    df = all_answers.reset_index()
    avg = df.groupby('index').mean()
    std = df.groupby('index').std()
    error_metric = AbsRelError()
    errors_df = error_metric.compute_errors_df(avg, gts_df)

    # all_xs, all_ys = [], []
    for col in errors_df.columns:
        xs = errors_df[col][~std[col].isna()]
        ys = std[col][~std[col].isna()]
        r,p = pearsonr(xs, ys)
        print(f"{col:<30}, {r:.3f}, {p:.3f}")
        # all_xs.extend(list(xs))
        # all_ys.extend(list(ys))


##### Checking correlations

for llm_name in _LLM_DICT:
    answers_df, gts_df = collect_responses_and_gts(llm_name = llm_name)
    error_metric = AbsRelError()
    errors_df = error_metric.compute_errors_df(answers_df, gts_df)
    avg = errors_df.mean(1)
    for q, init_query in _QUERY_DICT.items():
        query = init_query()
        gt_df = query.gt_df.set_index('Country Name')
        xs = avg[avg.index.isin(gt_df.index)]
        ys = gt_df.loc[xs.index].gt_answer
        xs = xs[~ys.isna()]
        r,p = pearsonr(xs, ys)
        print(f"{q:<40}, {r:.3f}, {p:.3f}")
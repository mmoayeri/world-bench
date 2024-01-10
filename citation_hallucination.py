

"""
Our goal here is to collect instances where the LLM cites a source, and checking
if the citation is accurate.

Our hypothesis is that the LLM may be hallucinating. 
"""

def extract_cited_year_and_gt_ans(ans, gt_row):
    # clean answer
    strs_to_remove = ['\n',')','(','<s>','</s>', ',', '.']
    for s in strs_to_remove:
        ans = ans.replace(s, ' ')
    
    words = ans.split(' ')
    year, gt_ans = '', np.NaN
    for w in words:
        try:
            if np.abs(float(w) - 2020) < 10:
                year = w
                gt_ans = gt_row.loc[year]
        except:
            continue
    return year, gt_ans


def find_hallucinations():
    cats = pd.read_csv('gt_answers/country_categories/region_and_income.csv')
    to_skip = [c.replace('\\', '').replace('\&', '&') for c in cats.iloc[220:].Economy] + ['Switzerland']

    years = range(2015, 2023)
    df = pd.DataFrame(columns=['llm_name','query_name','country','answer', 'parsed_answer', 'cited_year', 'gt_cited_year']+[f'gt_{year}' for year in years])

    error = AbsRelError()

    for llm_name, (make_llm, llm_key) in _LLM_DICT.items():
        llm = make_llm(llm_key)
        for query_name, init_query in _QUERY_DICT.items():
            query = init_query()
            gt_df = pd.read_csv(query.gt_csv_path, skiprows=[0,1,2,3]).set_index('Country Name')
            answers = load_cached_data(os.path.join(_CACHED_DATA_ROOT, 'answers_trial1', query_name, llm.get_modelname()), mode='json')['answers']
            for c, ans in answers.items():
                if c in to_skip:
                    continue
                gt_row = gt_df.loc[c]
                if 'world bank' in ans.lower(): # cited
                    parsed_answer = to_num(parse(ans))
                    row = [llm_name, query_name, c, ans, parsed_answer]
                    year, gt_ans = extract_cited_year_and_gt_ans(ans, gt_row)
                    row.extend([year, gt_ans])
                    for year in years:
                        row.append(gt_row[str(year)])
                    df.loc[len(df)] = row
            
    df['error_w_cited_year'] = df.apply(lambda row: error.error(row.parsed_answer, row.gt_cited_year), axis=1)
    df['error_w_any_year'] = df.apply(lambda row: np.min([error.error(row['parsed_answer'], row[f'gt_{year}']) 
                                                            for year in years if row[f'gt_{year}'] is not np.nan]), axis=1)

    print('Error compared to cited year: ')
    return df

def cited_vs_not_error():
    gaps_df = pd.DataFrame(columns=['Query', 'llm', 'c'])
    for llm_name, (make_llm, llm_key) in _LLM_DICT.items():
        llm = make_llm(llm_key)
        answers_df, gts_df = collect_responses_and_gts(llm_name=llm_name)
        errors_df = error_metric().compute_errors_df(answers_df, gts_df)
        for query_name in _QUERY_DICT:
            answers = load_cached_data(os.path.join(_CACHED_DATA_ROOT, 'answers_trial1', query_name, llm.get_modelname()), mode='json')['answers']
            errors_df[f'cited_{query_name}'] = [((c in answers) and ('world bank' in answers[c].lower())) for c in errors_df.index]
            print(errors_df.groupby(f'cited_{query_name}').mean(query_name))


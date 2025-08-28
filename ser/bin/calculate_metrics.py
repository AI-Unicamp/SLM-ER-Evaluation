import argparse
import pandas as pd
from scipy.stats import chi2_contingency, contingency
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--agg_results', type=str)
parser.add_argument('--out_file', type=str)
args = parser.parse_args()

df = pd.read_csv(args.agg_results)
df = df[df['pred_emo'].isin(['angry', 'happy', 'neutral', 'sad'])]

assert set(df['pred_emo'].tolist()) == set(df['txt_emo'].tolist()) == set(df['wav_emo'].tolist())

def calculate_metrics(x):
    acc = accuracy_score(x['wav_emo'], x['pred_emo'])
    f1 = f1_score(x['wav_emo'], x['pred_emo'], average='macro')

    err = x[x['wav_emo'] != x['pred_emo']]
    err_txt = err[err['txt_emo'] == err['pred_emo']]
    pct_txt_err = len(err_txt) / len(err)

    err_txt_implicit = err[(err['txt_emo'] == err['pred_emo']) & (err['txt_cond'] == 'implicit')]
    err_txt_explicit = err[(err['txt_emo'] == err['pred_emo']) & (err['txt_cond'] == 'explicit')]
    pct_txt_err_implicit = len(err_txt_implicit) / len(err)
    pct_txt_err_explicit = len(err_txt_explicit) / len(err)

    return pd.Series({
        'accuracy': acc,
        'f1_macro': f1,
        'pct_txt_pred_error': pct_txt_err,
        'pct_txt_pred_error_implicit': pct_txt_err_implicit,
        'pct_txt_pred_error_explicit': pct_txt_err_explicit
    })


results_df = df.groupby(['slm', 'tts_model']).apply(
    lambda x: calculate_metrics(x)
)

results_df['accuracy'] = results_df['accuracy'].map('{:.2%}'.format)
results_df['f1_macro'] = results_df['f1_macro'].map('{:.2f}'.format)
results_df['pct_txt_pred_error'] = results_df['pct_txt_pred_error'].map('{:.2%}'.format)
results_df['pct_txt_pred_error_implicit'] = results_df['pct_txt_pred_error_implicit'].map('{:.2%}'.format)
results_df['pct_txt_pred_error_explicit'] = results_df['pct_txt_pred_error_explicit'].map('{:.2%}'.format)

results_df.to_csv(args.out_file)

### Chi squared tests ###

cont_table_pred_wav = pd.crosstab(df['pred_emo'], df['wav_emo'])
cont_table_pred_txt = pd.crosstab(df['pred_emo'], df['txt_emo'])
print(cont_table_pred_txt)

chi2, p_value, dof, expected = chi2_contingency(cont_table_pred_wav)
cramers_v = contingency.association(cont_table_pred_wav, method='cramer')
print(f'Chi squared statistic (pred_emo X wav_emo):\nchi2 = {chi2} | p = {p_value} | v (effect size) = {cramers_v}\n')
chi2, p_value, dof, expected = chi2_contingency(cont_table_pred_txt)
cramers_v = contingency.association(cont_table_pred_txt, method='cramer')
print(f'Chi squared statistic (pred_emo X txt_emo):\nchi2 = {chi2} | p = {p_value} | v (effect size) = {cramers_v}\n')
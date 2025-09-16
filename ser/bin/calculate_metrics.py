import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    acc_proxy = accuracy_score(x['txt_emo'], x['pred_emo'])
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
        'accuracy_proxy': acc_proxy
    })


results_df = df.groupby(['slm', 'tts_model', 'txt_cond']).apply(
    lambda x: calculate_metrics(x)
)

# if args.out_file is not None:
#     results_df.to_csv(args.out_file)

### Chi squared tests ###

cont_table_pred_wav = pd.crosstab(df['pred_emo'], df['wav_emo'])
cont_table_pred_txt = pd.crosstab(df['pred_emo'], df['txt_emo'])
print(len(df))
sample_size = cont_table_pred_txt.sum().sum()
print('Contingengy table (Pred Emo X Txt Emo):\n', cont_table_pred_txt)
print('\nContingengy table (Pred Emo X Wav Emo):\n', cont_table_pred_wav)

chi2, p_value, dof, expected = chi2_contingency(cont_table_pred_wav)
cramers_v = contingency.association(cont_table_pred_wav, method='cramer')
print(f'Chi squared statistic (pred_emo X wav_emo):\nchi2 = {chi2} | p = {p_value} | v (effect size) = {cramers_v} | DoF = {dof} | Sample size = {sample_size}\n')

chi2, p_value, dof, expected = chi2_contingency(cont_table_pred_txt)
cramers_v = contingency.association(cont_table_pred_txt, method='cramer')
print(f'Chi squared statistic (pred_emo X txt_emo):\nchi2 = {chi2} | p = {p_value} | v (effect size) = {cramers_v} | DoF = {dof} | Sample size = {sample_size}\n')


### Heatmap ###
df_cong = df[df['lbl_full'].str.contains('cong')]
df_proxy = df[df['lbl_full'].str.contains('proxy')]

df_cong_qwen = df[(df['lbl_full'].str.contains('cong')) & (df['slm'] == 'qwen') & (df['txt_cond'] == 'neutral')]
df_proxy_qwen = df[(df['lbl_full'].str.contains('proxy')) & (df['slm'] == 'qwen') & (df['txt_cond'] == 'neutral')]

df_cong_salmonn = df[(df['lbl_full'].str.contains('cong')) & (df['slm'] == 'salmonn') & (df['txt_cond'] == 'neutral')]
df_proxy_salmonn = df[(df['lbl_full'].str.contains('proxy')) & (df['slm'] == 'salmonn') & (df['txt_cond'] == 'neutral')]

df_cong_desta = df[(df['lbl_full'].str.contains('cong')) & (df['slm'] == 'desta2') & (df['txt_cond'] == 'neutral')]
df_proxy_desta = df[(df['lbl_full'].str.contains('proxy')) & (df['slm'] == 'desta2') & (df['txt_cond'] == 'neutral')]

###
plt.figure(figsize=(6, 6))
freq_table_cong = pd.crosstab(df_cong['pred_emo'], df_cong['wav_emo'], normalize='columns')
annot_labels = freq_table_cong.applymap(lambda x: f'{(x*100):.1f}')
sns.heatmap(
    freq_table_cong, 
    #annot=True,
    annot_kws={"size": 20, "weight": "bold", "color": "black"},
    fmt='',
    annot=annot_labels,
    cmap='Blues',
    cbar=True
)
#plt.title('Congruent condition', fontsize=25)
plt.xlabel('Speech Emotion', fontsize=25)
plt.ylabel('Predicted Emotion', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("ser/bin/inference/v4_heatmap_cong_colNorm.pdf", bbox_inches='tight', pad_inches=0)
plt.close()

plt.figure(figsize=(6, 6))
freq_table_proxy = pd.crosstab(df_proxy['pred_emo'], df_proxy['wav_emo'], normalize='columns')
annot_labels = freq_table_proxy.applymap(lambda x: f'{(x*100):.1f}')
sns.heatmap(
    freq_table_proxy, 
    #annot=True,
    annot_kws={"size": 20, "weight": "bold", "color": "black"},
    fmt='',
    annot = annot_labels,
    cmap='Blues',
    cbar=True
)
#plt.title('Incongruent (proxy) condition', fontsize=20)
plt.xlabel('Speech Emotion', fontsize=25)
plt.ylabel('Predicted Emotion', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("ser/bin/inference/v4_heatmap_incong_colNorm.pdf", bbox_inches='tight', pad_inches=0)
plt.close()
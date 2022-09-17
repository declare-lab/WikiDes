# https://towardsdatascience.com/how-to-compare-two-or-more-distributions-9b06ee4d30bf
from read_write_file import *
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

from joypy import joyplot

def visualize_AB(input_file1 = 'dataset/phrase2/generated_test_para_256_diff_sim_best.json', \
                 input_file2 = 'dataset/phrase2/generated_test_para_256_diff_sim_best.json'):

    dataset = load_list_from_jsonl_file(input_file1)

    data_list = []
    for item in dataset:

        source_values = [v for k, v in item['source_dict'].items()]
        #print('source_values: ', source_values)
        data_list.append(['source'] + source_values)
        
        candidate_values = [v for k, v in item['candidate_dict'].items()]
        #print('candidate_values: ', candidate_values)
        data_list.append(['candidate'] + candidate_values)

        target_values = [v for k, v in item['target_dict'].items()]
        #print('target_values: ', target_values)
        data_list.append(['target'] + target_values)
        
    # create dataframe
    columns = ['Group', 'Negative', 'Neutral', 'Positive']
    df1 = pd.DataFrame(data_list, columns=columns)

    #-------------
    dataset = load_list_from_jsonl_file(input_file2)

    data_list = []
    for item in dataset:

        source_values = [v for k, v in item['source_dict'].items()]
        #print('source_values: ', source_values)
        data_list.append(['source'] + source_values)
        
        candidate_values = [v for k, v in item['candidate_dict'].items()]
        #print('candidate_values: ', candidate_values)
        data_list.append(['candidate'] + candidate_values)

        target_values = [v for k, v in item['target_dict'].items()]
        #print('target_values: ', target_values)
        data_list.append(['target'] + target_values)
        
    # create dataframe
    columns = ['Group', 'Negative', 'Neutral', 'Positive']
    df2 = pd.DataFrame(data_list, columns=columns)

    f, axes = plt.subplots(2, 3)
    
    sns.kdeplot(x='Negative', data=df1, hue='Group', fill=True, common_norm=False, ax=axes[0][0])
    #plt.title("Kernel Density")
    #plt.show()

    sns.kdeplot(x='Neutral', data=df1, hue='Group', fill=True, common_norm=False, ax=axes[0][1])
    #plt.title("Kernel Density")
    #plt.show()

    sns.kdeplot(x='Positive', data=df1, hue='Group', fill=True, common_norm=False, ax=axes[0][2])
    #plt.title("Kernel Density")

    sns.kdeplot(x='Negative', data=df2, hue='Group', fill=True, common_norm=False, ax=axes[1][0])
    #plt.title("Kernel Density")
    #plt.show()

    sns.kdeplot(x='Neutral', data=df2, hue='Group', fill=True, common_norm=False, ax=axes[1][1])
    #plt.title("Kernel Density")
    #plt.show()

    sns.kdeplot(x='Positive', data=df2, hue='Group', fill=True, common_norm=False, ax=axes[1][2])
    #plt.title("Kernel Density")
    
    plt.show()


    '''sns.histplot(x='pos', data=df, hue='group', bins=len(df), stat="density",
                 element="step", fill=False, cumulative=True, common_norm=False)
    plt.title("Cumulative Distribution")
    plt.show()'''

    
    '''joyplot(df, by='group', column='neg', colormap=sns.color_palette("crest", as_cmap=True));
    plt.xlabel('neu');
    plt.title("Ridgeline Plot, multiple groups")
    plt.show()'''


def qq_plot(input_file = 'dataset/phrase2/generated_test_para_256_diff_sim_best.json'):

    dataset = load_list_from_jsonl_file(input_file)

    data_list = []
    for item in dataset:

        source_values = [v for k, v in item['source_dict'].items()]
        #print('source_values: ', source_values)
        data_list.append(['source'] + source_values)
        
        candidate_values = [v for k, v in item['candidate_dict'].items()]
        #print('candidate_values: ', candidate_values)
        data_list.append(['candidate'] + candidate_values)

        target_values = [v for k, v in item['target_dict'].items()]
        #print('target_values: ', target_values)
        data_list.append(['target'] + target_values)

    # create dataframe
    columns = ['group', 'neg', 'neu', 'pos']
    df = pd.DataFrame(data_list, columns=columns)
    

    income = df['neu'].values
    income_t = df.loc[df.group=='source', 'neu'].values
    income_c = df.loc[df.group=='candidate', 'neu'].values

    df_pct = pd.DataFrame()
    df_pct['q_source'] = np.percentile(income_t, range(100))
    df_pct['q_candidate'] = np.percentile(income_c, range(100))

    plt.figure(figsize=(8, 8))
    plt.scatter(x='q_candidate', y='q_source', data=df_pct, label='Actual fit');
    sns.lineplot(x='q_source', y='q_source', data=df_pct, color='r', label='Line of perfect fit');
    plt.xlabel('Quantile of income, control group')
    plt.ylabel('Quantile of income, treatment group')
    plt.legend()
    plt.title("QQ plot")
    plt.show()

    

def calculate_difference(input_file = 'dataset/phrase2/generated_test_para_256_diff_sim_best.json'):

    dataset = load_list_from_jsonl_file(input_file)

    data_list = []
    for item in dataset:

        source_values = [v for k, v in item['source_dict'].items()]
        #print('source_values: ', source_values)
        data_list.append(['source'] + source_values)
        
        candidate_values = [v for k, v in item['candidate_dict'].items()]
        #print('candidate_values: ', candidate_values)
        data_list.append(['candidate'] + candidate_values)

        target_values = [v for k, v in item['target_dict'].items()]
        #print('target_values: ', target_values)
        data_list.append(['target'] + target_values)

    # create dataframe
    columns = ['group', 'neg', 'neu', 'pos']
    df = pd.DataFrame(data_list, columns=columns)

    # statistics
    
    labels = ['neg', 'neu', 'pos']

    for l in labels:
        sent_s = df.loc[df.group=='source', l].values
        sent_c = df.loc[df.group=='candidate', l].values
        sent_t = df.loc[df.group=='target', l].values

        stat, p_value = ttest_ind(sent_s, sent_c)
        print(f"t-test: statistic={stat:.4f}, p-value={p_value:.4f}")

        stat, p_value = ttest_ind(sent_t, sent_c)
        print(f"t-test: statistic={stat:.4f}, p-value={p_value:.4f}")

        stat, p_value = mannwhitneyu(sent_s, sent_c)
        print(f" Mann–Whitney U Test: statistic={stat:.4f}, p-value={p_value:.4f}")

        stat, p_value = mannwhitneyu(sent_t, sent_c)
        print(f" Mann–Whitney U Test: statistic={stat:.4f}, p-value={p_value:.4f}")

        '''all_sent = df[l].values
        sample_stat = np.mean(sent_s) - np.mean(sent_c)
        stats = np.zeros(1000)
        for k in range(1000):
            labels = np.random.permutation((df['group'] == 'candidate').values)
            stats[k] = np.mean(all_sent[labels]) - np.mean(all_sent[labels==False])
        p_value = np.mean(stats > sample_stat)
        print(f"Permutation test: p-value={p_value:.4f}")

        sample_stat = np.mean(sent_t) - np.mean(sent_c)
        stats = np.zeros(1000)
        for k in range(1000):
            labels = np.random.permutation((df['group'] == 'candidate').values)
            stats[k] = np.mean(all_sent[labels]) - np.mean(all_sent[labels==False])
        p_value = np.mean(stats > sample_stat)
        print(f"Permutation test: p-value={p_value:.4f}")'''

def calculate_ks(input_file = 'dataset/phrase2/generated_test_para_256_diff_sim_best.json'):

    dataset = load_list_from_jsonl_file(input_file)

    data_list = []
    for item in dataset:

        source_values = [v for k, v in item['source_dict'].items()]
        #print('source_values: ', source_values)
        data_list.append(['source'] + source_values)
        
        candidate_values = [v for k, v in item['candidate_dict'].items()]
        #print('candidate_values: ', candidate_values)
        data_list.append(['candidate'] + candidate_values)

        target_values = [v for k, v in item['target_dict'].items()]
        #print('target_values: ', target_values)
        data_list.append(['target'] + target_values)

    # create dataframe
    columns = ['group', 'neg', 'neu', 'pos']
    df = pd.DataFrame(data_list, columns=columns)

    # statistics
    
    labels = ['neg', 'neu', 'pos']

    f, axes = plt.subplots(3, 2)
    
    #sns.kdeplot(x='Negative', data=df1, hue='Group', fill=True, common_norm=False, ax=axes[0][0])


    for i, l in enumerate(labels):
        print(i, l)
        sent_s = df.loc[df.group=='source', l].values
        sent_c = df.loc[df.group=='candidate', l].values
        sent_t = df.loc[df.group=='target', l].values

        df_ks = pd.DataFrame()  
        df_ks['F_' + l] = np.sort(df[l].unique())
        df_ks['F_control'] = df_ks['F_' + l].apply(lambda x: np.mean(sent_s<=x))
        df_ks['F_treatment'] = df_ks['F_' + l].apply(lambda x: np.mean(sent_c<=x))

        k = np.argmax(np.abs(df_ks['F_control'] - df_ks['F_treatment']))
        ks_stat = np.abs(df_ks['F_treatment'][k] - df_ks['F_control'][k])
        

        y = (df_ks['F_treatment'][k] + df_ks['F_control'][k])/2
        axes[i][0].plot('F_' + l, 'F_control', data=df_ks, label='source')
        axes[i][0].plot('F_' + l, 'F_treatment', data=df_ks, label='candidate')
        axes[i][0].errorbar(x=df_ks['F_' + l][k], y=y, yerr=ks_stat/2, color='k',
             capsize=5, mew=3, label=f"Test statistic: {ks_stat:.4f}")

        if (l == 'neu'):
            axes[i][0].legend(loc='center left')
        else:
            axes[i][0].legend(loc='center right')
        #axes[i][0].title(l)

        #----------------
        df_ks = pd.DataFrame()  
        df_ks['F_' + l] = np.sort(df[l].unique())
        df_ks['F_control'] = df_ks['F_' + l].apply(lambda x: np.mean(sent_t<=x))
        df_ks['F_treatment'] = df_ks['F_' + l].apply(lambda x: np.mean(sent_c<=x))

        k = np.argmax(np.abs(df_ks['F_control'] - df_ks['F_treatment']))
        ks_stat = np.abs(df_ks['F_treatment'][k] - df_ks['F_control'][k])

        y = (df_ks['F_treatment'][k] + df_ks['F_control'][k])/2
        axes[i][1].plot('F_' + l, 'F_control', data=df_ks, label='target')
        axes[i][1].plot('F_' + l, 'F_treatment', data=df_ks, label='candidate')
        axes[i][1].errorbar(x=df_ks['F_' + l][k], y=y, yerr=ks_stat/2, color='k',
             capsize=5, mew=3, label=f"Test statistic: {ks_stat:.4f}")

        if (l == 'neu'):
            axes[i][1].legend(loc='center left')
        else:
            axes[i][1].legend(loc='center right')
        #axes[i][1].title(l)
        #plt.show()

    plt.show()

def calculate_stats(input_file = ''):

    dataset = load_list_from_jsonl_file(input_file)

    data_list = []
    for item in dataset:

        source_values = [v for k, v in item['source_dict'].items()]
        #print('source_values: ', source_values)
        data_list.append(['source'] + source_values)
        
        candidate_values = [v for k, v in item['candidate_dict'].items()]
        #print('candidate_values: ', candidate_values)
        data_list.append(['candidate'] + candidate_values)

        target_values = [v for k, v in item['target_dict'].items()]
        #print('target_values: ', target_values)
        data_list.append(['target'] + target_values)

    # create dataframe
    columns = ['group', 'neg', 'neu', 'pos']
    df = pd.DataFrame(data_list, columns=columns)

    labels = ['neg', 'neu', 'pos']

    for label in labels:
        values = df.loc[df.group=='source', label].values
        print(label + '_source: ', max(values), min(values), sum(values)/len(values))

        values = df.loc[df.group=='candidate', label].values
        print(label + '_candidate: ', max(values), min(values), sum(values)/len(values))

        values = df.loc[df.group=='target', label].values
        print(label + '_target: ', max(values), min(values), sum(values)/len(values))
        print('----------------')

    print('....................')
        
#....................
if __name__ == '__main__':

    #qq_plot(input_file = 'dataset/phrase2/generated_test_para_256_diff_sim_best.json')
    #calculate_difference(input_file = 'dataset/phrase2/generated_test_para_256_diff_sim_best.json')

    #calculate_stats(input_file = 'dataset/phrase2/generated_test_para_256_diff_simrouge_best.json')
    #calculate_stats(input_file = 'dataset/phrase2/generated_test_para_256_random_rouge_best.json')
    
    visualize_AB(input_file1 = 'dataset/phrase2/generated_test_para_256_diff_simrouge_best.json', \
                 input_file2 = 'dataset/phrase2/generated_test_para_256_random_rouge_best.json')


    
    #calculate_ks(input_file = 'dataset/phrase2/generated_test_para_256_diff_simrouge_best.json')
    #calculate_ks(input_file = 'dataset/phrase2/generated_test_para_256_random_rouge_best.json')

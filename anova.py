import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

def anova_analysis(df, dependent_variable, columns_to_compare):
    """
    Parameters:
    - depent_variable (str) : The column name of the dependent variable. 
        - one-way : one dependent variable. (disease cohort)
    - columns_to_compare (list) : List of columns name to compare using ANOVA. (acceleration, residual)

    Returns:
    - anova_result (pd.DataFrame) : The result of ANOVA analysis. (contains F statistics, p-value)
    """
    category_list = list(df[dependent_variable].unique())
    group_indices = {category: list(df[df[dependent_variable] == category].index) for category in category_list}
  
    namee = []
    pva = []
    stata = []
    group_medians = {category: [] for category in category_list}
    group_se = {category: [] for category in category_list}
    
    for col in columns_to_compare:
        groups = [df.loc[group_indices[category], col].dropna().values for category in category_list]
        [aov_stat, aov_pval] = f_oneway(*groups)
        for category in category_list:
            group_medians[category].append(np.median([float(i) for i in groups[category_list.index(category)]]))
            group_se[category].append(np.std([float(i) for i in groups[category_list.index(category)]], ddof=1) / np.sqrt(len(groups[category_list.index(category)])))

        namee.append(col)
        pva.append(aov_pval)
        stata.append(aov_stat)

    df = pd.DataFrame(list(zip(stata, pva)), index=namee, columns=['statistic', 'pvalue'])

    # Correct p-values
    df['pvalue_corr'] = multipletests(df['pvalue'], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]

    for category in category_list:
        df[f'median: {category}'] = group_medians[category]
        df[f'se_{category}'] = group_se[category]

    return df

df1 = pd.read_csv('/home/dooharin/2024_ukb/data/senario 2-5/statistical analyses/residuals/overall_acc/asthma_disease_res.csv')
df2 = pd.read_csv('/home/dooharin/2024_ukb/data/senario 2-5/statistical analyses/residuals/overall_acc/arrhythmia_disease_res.csv')
df3 = pd.read_csv('/home/dooharin/2024_ukb/data/senario 2-5/statistical analyses/residuals/overall_acc/panic_disease_res.csv')
df4 = pd.read_csv('/home/dooharin/2024_ukb/data/senario 2-5/statistical analyses/residuals/overall_acc/angina_disease_res.csv')
df5 = pd.read_csv('/home/dooharin/2024_ukb/data/senario 2-5/statistical analyses/residuals/overall_acc/migraine_disease_res.csv')
df6 = pd.read_csv('/home/dooharin/2024_ukb/data/senario 2-5/statistical analyses/residuals/overall_acc/hyperthyrodism_disease_res.csv')
df7 = pd.read_csv('/home/dooharin/2024_ukb/data/senario 2-5/statistical analyses/residuals/overall_acc/hypoglycemia_disease_res.csv')
df8 = pd.read_csv('/home/dooharin/2024_ukb/data/senario 2-5/statistical analyses/residuals/overall_acc/vestibular_disease_res.csv')

merged_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])
df = merged_df.copy()

dependent_variable = 'group'
columns_to_compare = ['mean_acceleration', 'average acceleration residuals']  
anova_df = anova_analysis(df, dependent_variable, columns_to_compare)

anova_df

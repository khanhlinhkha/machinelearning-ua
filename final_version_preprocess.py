#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

#Import the dataset
df_ = pd.read_csv("../../Downloads/conversion_DB_Train.csv", sep =",")
df = df_.copy()
df = df.drop('availableActions', axis=1)
df = df.drop('isSecondHandVehicle', axis=1)

# DATA UNDERSTANDING

# Function to check the dataset
def check_df(dataframe, head=5):
    print('##################### Columns #####################')
    print(dataframe.columns)
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Number of null values #####################')
    print(dataframe.isnull().sum())
    print('##################### Total of null values #####################')
    print(dataframe.isnull().sum().sum())
    print('##################### Ratio of null values #####################')
    print((dataframe.isnull().sum())/len(dataframe))
    print('##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.75, 0.99, 1]).T)
    print('##################### Info #####################')
    print(dataframe.info())

check_df(df)

# Frequency distribution of target variable

print(df['converted'].value_counts())
print((df['converted'].value_counts())/len(df))
print(df['converted'].value_counts().plot(kind = 'bar'))
plt.xlabel("Converted or not")
plt.ylabel("Frequency")
plt.title("Frequency distribution of target variable")

# 0    0.650686 - 0
# 1    0.349314 - 1
# 0.65 & 0.35 distribution is slightly imbalanced so there is no need to use imbalance handling methods.

# Frequency distribution of NOMINAL and ORDINAL variables

categorial_variable = ['Year_Month', 'dero', 'animation','animation_2','profession', 'fuelType',
                       'make','model', 'mainDriverNotDesignated',
                       'postal_code_XX', 'premiumCustomer','customerScore']

def plot_bar_graphs_for_variables(dataframe, variable_names):
    for variable_name in variable_names:
        if variable_name in dataframe.columns:
            value_counts = dataframe[variable_name].value_counts(dropna=False)
            value_counts.plot(kind='bar', figsize=(8, 6))
            plt.xlabel(variable_name)
            plt.ylabel('Count')
            plt.title(f'Bar Graph for {variable_name}')
            plt.show()
plot_bar_graphs_for_variables(df, categorial_variable)

# Frequency distribution of DISCRETE variables

discrete_variable = ['nbYearsAttest', 'birthday_5y', 'license_year_5y','vhConstructionYear',
                     'availableActions','purchaseTva','nbbackoffice']

#Histogram
def plot_histograms_for_variables(dataframe, variable_names):
    for variable_name in variable_names:
        if variable_name in dataframe.columns:
            plt.figure(figsize=(8, 6))
            dataframe[variable_name].plot(kind='hist', bins=10)  # You can adjust the number of bins as needed
            plt.xlabel(variable_name)
            plt.ylabel('Frequency')
            plt.title(f'Histogram for {variable_name}')
            plt.show()

plot_histograms_for_variables(df, discrete_variable)

# Frequency distribution of CONTINUOUS variables

continuous_variable = ['powerKW', 'catalogValue']

# Boxplot
def plot_boxplots_for_variables(dataframe, variable_names):
    for variable_name in variable_names:
        if variable_name in dataframe.columns:
            plt.figure(figsize=(8, 6))
            dataframe.boxplot(column=variable_name)
            plt.xlabel('Variable')
            plt.ylabel('Value')
            plt.title(f'Boxplot for {variable_name}')
            plt.show()

plot_boxplots_for_variables(df,continuous_variable)

# Histogram
plot_histograms_for_variables(df, continuous_variable)

# Check missing values percentage

missing_ratio = (df.isnull().sum())/len(df)
sorted_missing_ratio = missing_ratio.sort_values(ascending=False)
print(sorted_missing_ratio)

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['number of missing values', 'percentage of missing values'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

na_cols = missing_values_table(df, True)


# Checking missing values vs Target to find out if there is a meaningfull correlation
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "converted", na_cols)

# Check outliers for continuous variables

columns_for_outlier_check = ['powerKW', 'catalogValue']

def outlier_thresholds(dataframe, col_name, q1=.25, q3=.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=.25, q3=.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        num_outliers = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].shape[0]
        print("Number of outliers in", col_name, "is:", num_outliers)
        return True
    else:
        return False

for elem in columns_for_outlier_check:
  check_outlier(df, elem)

# Correlation

drop_list =['Year_Month', 'postal_code_XX', 'mainDriverNotDesignated']
df_corr = df.drop(drop_list, axis=1)
corr = df_corr.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

def high_correlated_cols(dataframe,head=10):
    corr_matrix = dataframe.corr().abs()
    corr_cols = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                                   .astype(bool)).stack().sort_values(ascending=False)).head(head)
    return corr_cols

high_correlated_cols(df_corr)

# DATA PREPROCESSING

# Divide data into train/valid/test data

from sklearn.model_selection import train_test_split
indices=np.arange(44928)
indices_train, indices_test  = train_test_split(indices, test_size=0.2, random_state=0)
indices_train, indices_val  = train_test_split(indices_train, test_size=0.2, random_state=0)

# Replace missing values with mode of train set for dicrete & nominal; or mean of train set for continuous variables

def replace_missing_values_by_mode(df,var,train_ind):
    var_train = df[var].iloc[train_ind]
    var_train_mode = var_train.mode()[0]
    df[var].fillna(var_train_mode,inplace = True)

def replace_missing_values_by_mean(df,var,train_ind):
    var_train = df[var].iloc[train_ind]
    var_train_mean = np.mean(var_train)
    df[var].fillna(var_train_mean,inplace = True)

variables_fill_mode = ['nbYearsAttest','profession','license_year_5y','purchaseTva','fuelType',
                      'vhConstructionYear','birthday_5y']

for variable in variables_fill_mode:
    replace_missing_values_by_mode(df,variable,indices_train)

replace_missing_values_by_mean(df, 'catalogValue',indices_train)

# mainDriverNotDesignated
df['mainDriverNotDesignated'].fillna(True, inplace = True)

# To control
df.info()

df_copy = df.copy()

# Normalization and outliers treatment
def standardize(df, var, train_ind):
    var_train = df[var].iloc[train_ind]
    var_train_mean = np.mean(var_train)
    var_train_std = np.std(var_train)
    #Standardize variable
    df[var] = (df[var] - var_train_mean) / var_train_std
    return df[var]

def outliers_treatment(df, var, train_ind):
    df[var] = standardize(df,var,train_ind)
    #Deal with outliers
    df[var] = np.where(df[var] > 3, 3, df[var])
    df[var] = np.where(df[var] < -3, -3, df[var])
    return df

# powerKW -- continuous
outliers_treatment(df_copy,'powerKW',indices_train)

# catalogValue -- continuous
outliers_treatment(df_copy,'catalogValue',indices_train)

# Check outliers of discrete variables manually and eliminate them

# Birthday
df_copy['birthday_5y'].value_counts() # 1695.0 should be eliminated
birthday_5y_train = df_copy['birthday_5y'].iloc[indices_train]
birthday_5y_train_mode = birthday_5y_train.mode()[0]
df_copy['birthday_5y'] = df_copy['birthday_5y'].replace(1695,birthday_5y_train_mode)
df_copy['birthday_5y'].value_counts()

df_copy['license_year_5y'].value_counts() # seems no outlier

df_copy['vhConstructionYear'].value_counts() # 1014 should be eliminated since first car was build at 1886
vhConstructionYear_train = df_copy['vhConstructionYear'].iloc[indices_train]
vhConstructionYear_train_mode = vhConstructionYear_train.mode()[0]
df_copy['vhConstructionYear'] = df_copy['vhConstructionYear'].replace(1014,vhConstructionYear_train_mode)

# Adding new columns for date variables:

from datetime import datetime
current_date = datetime.now()
df_copy['NEWAgeOfVehicle'] = current_date.year - df_copy['vhConstructionYear']
df_copy['NEWAgeOfCustomer'] = current_date.year - df_copy['birthday_5y']
df_copy['NEWTotalLicenseYear'] = current_date.year - df_copy['license_year_5y']

df_copy['NEWAgeOfVehicle'].describe()
df_copy['NEWTotalLicenseYear'].describe()
df_copy['NEWAgeOfCustomer'].describe()

df_copy.loc[(df_copy["NEWAgeOfCustomer"] < 33), "NEWAgeCustInterval"] = 'young'
df_copy.loc[(df_copy["NEWAgeOfCustomer"] < 43) & (df_copy["NEWAgeOfCustomer"] >= 33), "NEWAgeCustInterval"] = 'young_middle'
df_copy.loc[(df_copy["NEWAgeOfCustomer"] < 58) & (df_copy["NEWAgeOfCustomer"] >= 43), "NEWAgeCustInterval"] = 'middle_old'
df_copy.loc[(df_copy["NEWAgeOfCustomer"] >= 58), "NEWAgeCustInterval"] = 'old'

df_copy.loc[(df_copy["NEWAgeOfVehicle"] < 1), "NEWAgeVehicleInterval"] = 'young'
df_copy.loc[(df_copy["NEWAgeOfVehicle"] < 5) & (df_copy["NEWAgeOfVehicle"] >= 1), "NEWAgeVehicleInterval"] = 'young_middle'
df_copy.loc[(df_copy["NEWAgeOfVehicle"] < 11) & (df_copy["NEWAgeOfVehicle"] >= 5), "NEWAgeVehicleInterval"] = 'middle'
df_copy.loc[(df_copy["NEWAgeOfVehicle"] >= 11), "NEWAgeVehicleInterval"] = 'old'

df_copy.head()
df_copy.isnull().any().sum()

# Correlation check
drop_list =['Year_Month', 'postal_code_XX', 'NEWAgeVehicleInterval','NEWAgeCustInterval']
df_copy_corr = df_copy.drop(drop_list, axis=1)
corr = df_copy_corr.corr()

sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

def high_correlated_cols(dataframe,head=10):
    corr_matrix = dataframe.corr().abs()
    corr_cols = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                                   .astype(bool)).stack().sort_values(ascending=False)).head(head)
    return corr_cols

high_correlated_cols(df_copy_corr)

df_copy_2 = df_copy.copy()

# Encoding & Binning

# Encoding function
def encoding(Var_pre, encoding_opt, Var_name, thermo_categories=[''], thermo_opt="normal"):
    if encoding_opt=='dummy':
        Var_preprocessed=pd.get_dummies(Var_pre,prefix= str(Var_name))
        Categories=list(Var_preprocessed.columns.values)

        Reference_Category = Categories[0]
        Var_preprocessed = Var_preprocessed.iloc[:, 1:]
        Categories = Categories[1:]

    elif (encoding_opt=='thermometer'):
        for j, value in enumerate(thermo_categories):
            Var_pre = Var_pre.replace(value,j)
        Categories = [Var_name + '_'+category for category in thermo_categories]
        Reference_Category = Categories[0]
        Categories = Categories[1:]
        Var_preprocessed = pd.DataFrame(
            data=(np.arange(Var_pre.max()) < np.array(Var_pre).reshape(-1, 1)).astype(int),
            columns=Categories,
            index=Var_pre.index)
    elif (encoding_opt == 'ordinal'):
        for j, value in enumerate(thermo_categories):
            Var_pre = Var_pre.replace(value,j)
        return Var_pre

    if encoding_opt in ['dummy','thermometer']:
        return (Var_preprocessed, Categories, Reference_Category)

# Dummy encoding:

def dummy_encoding(df,cols):
    for col in cols:
        var_encoded = encoding(df[col], 'dummy',col)
        df = pd.concat([df,var_encoded[0]],axis=1)
    return df

dummy_cols = ['Year_Month','nbYearsAttest','profession','fuelType','mainDriverNotDesignated',
        'purchaseTva','nbbackoffice', 'premiumCustomer', 'NEWAgeCustInterval', 'NEWAgeVehicleInterval']

df_copy_2 = dummy_encoding(df_copy_2,dummy_cols)
df_copy_2.head()

# animation_2 - thermometer
animation_2_encoded = encoding(df_copy_2['animation_2'],'thermometer','animation_2',['0','1','2'],'special')
animation_2_encoded[0]
df_copy_2 = pd.concat([df_copy_2,animation_2_encoded[0]],axis=1)
df_copy_2.info()

# animation - thermometer
animation_encoded = encoding(df_copy_2['animation'],'thermometer','animation',['0','1','2','3','4','5','6','7'],'special')
animation_encoded[0]
df_copy_2 = pd.concat([df_copy_2,animation_encoded[0]],axis=1)
df_copy_2.info()

# dero - thermometer
df['dero'].nunique() # 17
dero_encoded = encoding(df_copy_2['dero'],'thermometer','dero',['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'],'special')
dero_encoded[0]
df_copy_2 = pd.concat([df_copy_2,dero_encoded[0]],axis=1)
df_copy_2.info()

# First binning and then thermometer encoding
df_copy_2['customerScore_Binned'] = pd.qcut(df_copy_2['customerScore'], q=15, labels=False, duplicates='drop')
customerScore_Binned_encoded = encoding(df_copy_2['customerScore_Binned'],'thermometer','customerScore_Binned',['0','1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],'special')
customerScore_Binned_encoded[0]
df_copy_2 = pd.concat([df_copy_2,customerScore_Binned_encoded[0]],axis=1)

# equal frequency binning
# birthday_5y
df_copy_2['birthday_5y_binned'] = pd.qcut(df_copy_2['birthday_5y'], q=10, precision=0, retbins=False, duplicates='drop')
df_copy_2['birthday_5y_binned'].value_counts()

# license_year
df_copy_2['license_year_5y_binned'] = pd.qcut(df_copy_2['license_year_5y'], q=5, precision=0, retbins=False, duplicates='drop')
df_copy_2['license_year_5y_binned'].value_counts()

# vhConstructionYear
df_copy_2['vhConstructionYear_binned'] = pd.qcut(df_copy_2['vhConstructionYear'], q=13, precision=0, retbins=False, duplicates='drop')
df_copy_2['vhConstructionYear_binned'].value_counts()

# NEWAgeOfVehicle
df_copy_2['NEWAgeOfVehicle_Binned'] = pd.qcut(df_copy_2['NEWAgeOfVehicle'], q=13, precision=0, retbins=False, duplicates='drop')
df_copy_2['NEWAgeOfVehicle_Binned'].value_counts()

# NEWAgeOfCustomer
df_copy_2['NEWAgeOfCustomer_Binned'] = pd.qcut(df_copy_2['NEWAgeOfCustomer'], q=10, precision=0, retbins=False, duplicates='drop')
df_copy_2['NEWAgeOfCustomer_Binned'].value_counts()

df_copy_2['NEWTotalLicenseYear_Binned'] = pd.qcut(df_copy_2['NEWTotalLicenseYear'], q=5, precision=0, retbins=False, duplicates='drop')
df_copy_2['NEWTotalLicenseYear_Binned'].value_counts()

dummy_cols_2 = ['NEWTotalLicenseYear_Binned','birthday_5y_binned','license_year_5y_binned','vhConstructionYear_binned','NEWAgeOfVehicle_Binned', 'NEWAgeOfCustomer_Binned']
df_copy_2 = dummy_encoding(df_copy_2,dummy_cols_2)

df_copy_2.columns = df_copy_2.columns.str.replace('[\[\]\(\)]', '', regex=True)
df_copy_2.head()


value_counts = df_copy_2['make'].value_counts()
print(value_counts/len(df_copy_2))
value_counts = df_copy_2['model'].value_counts()
print(value_counts/len(df_copy_2))

# Make and model creating Others group
threshold = 0.05
value_counts = df_copy_2['make'].value_counts()
total_records = len(df_copy_2)
less_frequent_values = value_counts[value_counts / total_records < threshold].index
df_copy_2['make'] = df_copy_2['make'].apply(lambda x: 'Others' if x in less_frequent_values else x)

df_copy_2 = dummy_encoding(df_copy_2,['make'])

# Model
(df_copy_2['model'].value_counts())/len(df)

threshold = 0.01
value_counts = df_copy_2['model'].value_counts()
total_records = len(df_copy_2)
less_frequent_values = value_counts[value_counts / total_records < threshold].index
df_copy_2['model'] = df_copy_2['model'].apply(lambda x: 'Others' if x in less_frequent_values else x)

df_copy_2 = dummy_encoding(df_copy_2,['model'])

df_copy_2.head()

# WoE calculation for postal code:

import numpy as np

def calculate_woe(train_idx, test_idx, val_idx, feature, data, target):
    train = data.loc[train_idx]
    test = data.loc[test_idx]
    val = data.loc[val_idx]

    total_positive = train[target].sum()
    total_negative = train[target].count() - total_positive

    woe_dict = {}
    for category in train[feature].unique():
        category_positive = train.loc[train[feature] == category, target].sum()
        category_negative = train.loc[train[feature] == category, target].count() - category_positive

        if category_positive == 0:
            woe_dict[category] = np.log((1 / total_negative) / (1 / total_positive))
        elif category_negative == 0:
            woe_dict[category] = np.log((1 / total_negative) / (1 / total_positive))
        else:
            woe = np.log((category_negative / total_negative) / (category_positive / total_positive))

            if np.isinf(woe) or np.isnan(woe):
                woe_dict[category] = np.nan
            else:
                woe_dict[category] = woe

    data[f'{feature}_WoE'] = data[feature].map(woe_dict)
    train[f'{feature}_WoE'] = train[feature].map(woe_dict)
    test[f'{feature}_WoE'] = test[feature].map(woe_dict)
    val[f'{feature}_WoE'] = val[feature].map(woe_dict)

    return train, test, val

for feature in ['postal_code_XX']:
    train_data, test_data, validation_data = calculate_woe(
        indices_train, indices_test, indices_val, feature, df_copy_2, 'converted'
    )

df_copy_2.head()

# Drop columns

column_to_drop = ['Unnamed: 0', 'license_year_5y','Year_Month','converted','dero','animation','animation_2',
                  'nbYearsAttest','profession', 'birthday_5y','fuelType',
                  'vhConstructionYear', 'mainDriverNotDesignated','purchaseTva', 'customerScore', 'model', 'make', 'postal_code_XX',
                  'nbbackoffice','premiumCustomer', 'NEWAgeOfVehicle',	'NEWAgeOfCustomer',	'NEWTotalLicenseYear',	'NEWAgeCustInterval',	'NEWAgeVehicleInterval',
                  'birthday_5y_binned',	'license_year_5y_binned','customerScore_Binned', 'vhConstructionYear_binned','NEWAgeOfVehicle_Binned',	'NEWAgeOfCustomer_Binned',	'NEWTotalLicenseYear_Binned']

df_copy_3 = df_copy_2.drop(column_to_drop, axis = 1)

converted = df_copy_2['converted']

import openpyxl
with pd.ExcelWriter('Preprocessed_data.xlsx') as writer:
    df_copy_3.to_excel(writer, sheet_name='Data preprocessed')
    converted.to_excel(writer, sheet_name='converted')
    pd.DataFrame(indices_test).to_excel(writer, sheet_name='indices test')
    pd.DataFrame(indices_train).to_excel(writer, sheet_name='indices train')
    pd.DataFrame(indices_val).to_excel(writer, sheet_name='indices val')

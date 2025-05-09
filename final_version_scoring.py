import pandas as pd
import numpy as np

file_train = 'conversion_DB_Train.csv'
file_test = 'conversion_DB_Test_231114.csv'

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

df_train = pd.read_csv(file_train)
df_test = pd.read_csv(file_test)
print(df_train.columns)
print(df_test.columns)

df_train = df_train.drop('availableActions', axis=1)
df_train = df_train.drop('isSecondHandVehicle', axis=1)
df_train = df_train.drop('Unnamed: 0', axis=1)
df_test = df_test.drop('isSecondHandVehicle', axis=1)
df_test = df_test.drop('ID', axis=1)

print(len(df_test.columns))
print(len(df_train.columns))

# Merge the two DataFrames
merged_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
train_indices = list(range(0, 44928))
test_indices = list(range(44928, 49928))

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['number of missing values', 'percentage of missing values'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(merged_df)

# PREPROCESSING:

def replace_missing_values_by_mode(df,var,train_ind):
    var_train = df[var].iloc[train_ind]
    var_train_mode = var_train.mode()[0]
    df[var].fillna(var_train_mode,inplace = True)

def replace_missing_values_by_mean(df,var,train_ind):
    var_train = df[var].iloc[train_ind]
    var_train_mean = np.mean(var_train)
    df[var].fillna(var_train_mean,inplace = True)

variables_fill_mode = ['nbYearsAttest','profession','license_year_5y', 'purchaseTva','fuelType',
                      'vhConstructionYear','birthday_5y']

for variable in variables_fill_mode:
    replace_missing_values_by_mode(merged_df,variable,train_indices)

replace_missing_values_by_mean(merged_df, 'catalogValue', train_indices)
replace_missing_values_by_mean(merged_df, 'powerKW', train_indices)

# mainDriverNotDesignated
merged_df['mainDriverNotDesignated'].fillna(True, inplace = True)

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
outliers_treatment(merged_df,'powerKW',train_indices)

# catalogValue -- continuous
outliers_treatment(merged_df,'catalogValue',train_indices)

# Birthday
merged_df['birthday_5y'].value_counts() # 1695.0 should be eliminated
birthday_5y_train = merged_df['birthday_5y'].iloc[train_indices]
birthday_5y_train_mode = birthday_5y_train.mode()[0]
merged_df['birthday_5y'] = merged_df['birthday_5y'].replace(1695,birthday_5y_train_mode)

#License_Year
merged_df['license_year_5y'].value_counts() # seems no outlier

# vhConstructionYear
merged_df['vhConstructionYear'].value_counts() # 1014 should be eliminated since first car was build at 1886
vhConstructionYear_train = merged_df['vhConstructionYear'].iloc[train_indices]
vhConstructionYear_train_mode = vhConstructionYear_train.mode()[0]
merged_df['vhConstructionYear'] = merged_df['vhConstructionYear'].replace(1014,vhConstructionYear_train_mode)

# Adding new columns for date variables:

from datetime import datetime
current_date = datetime.now()
merged_df['NEWAgeOfVehicle'] = current_date.year - merged_df['vhConstructionYear']
merged_df['NEWAgeOfCustomer'] = current_date.year - merged_df['birthday_5y']
merged_df['NEWTotalLicenseYear'] = current_date.year - merged_df['license_year_5y']

merged_df.loc[test_indices]['NEWAgeOfCustomer'].describe()

merged_df['NEWAgeOfVehicle'].describe()
merged_df['NEWTotalLicenseYear'].describe()
merged_df['NEWAgeOfCustomer'].describe()

merged_df.loc[(merged_df["NEWAgeOfVehicle"] < 1), "NEWAgeVehicleInterval"] = 'young'
merged_df.loc[(merged_df["NEWAgeOfVehicle"] < 5) & (merged_df["NEWAgeOfVehicle"] >= 1), "NEWAgeVehicleInterval"] = 'young_middle'
merged_df.loc[(merged_df["NEWAgeOfVehicle"] < 11) & (merged_df["NEWAgeOfVehicle"] >= 5), "NEWAgeVehicleInterval"] = 'middle'
merged_df.loc[(merged_df["NEWAgeOfVehicle"] >= 11), "NEWAgeVehicleInterval"] = 'old'

merged_df.loc[(merged_df["NEWAgeOfCustomer"] < 33), "NEWAgeCustInterval"] = 'young'
merged_df.loc[(merged_df["NEWAgeOfCustomer"] < 43) & (merged_df["NEWAgeOfCustomer"] >= 33), "NEWAgeCustInterval"] = 'young_middle'
merged_df.loc[(merged_df["NEWAgeOfCustomer"] < 58) & (merged_df["NEWAgeOfCustomer"] >= 43), "NEWAgeCustInterval"] = 'middle_old'
merged_df.loc[(merged_df["NEWAgeOfCustomer"] >= 58), "NEWAgeCustInterval"] = 'old'

merged_df.head()
merged_df.isnull().any().sum()

merged_df.isnull().any()

# Encoding & Binning
merged_df_2 = merged_df.copy()

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

merged_df_2 = dummy_encoding(merged_df_2,dummy_cols)
merged_df_2.head()

# animation_2 - thermometer
animation_2_encoded = encoding(merged_df_2['animation_2'],'thermometer','animation_2',['0','1','2'],'special')
animation_2_encoded[0]
merged_df_2 = pd.concat([merged_df_2,animation_2_encoded[0]],axis=1)

# animation - thermometer
animation_encoded = encoding(merged_df_2['animation'],'thermometer','animation',['0','1','2','3','4','5','6','7'],'special')
animation_encoded[0]
merged_df_2 = pd.concat([merged_df_2,animation_encoded[0]],axis=1)

# dero - thermometer
merged_df_2['dero'].nunique() # 17
dero_encoded = encoding(merged_df_2['dero'],'thermometer','dero',['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'],'special')
dero_encoded[0]
merged_df_2 = pd.concat([merged_df_2,dero_encoded[0]],axis=1)
merged_df_2.info()

# First binning and then thermometer encoding
merged_df_2['customerScore_Binned'] = pd.qcut(merged_df_2['customerScore'], q=15, labels=False, duplicates='drop')
customerScore_Binned_encoded = encoding(merged_df_2['customerScore_Binned'],'thermometer','customerScore_Binned',['0','1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],'special')
customerScore_Binned_encoded[0]
merged_df_2 = pd.concat([merged_df_2,customerScore_Binned_encoded[0]],axis=1)

# equal frequency binning
# birthday_5y
merged_df_2['birthday_5y_binned'] = pd.qcut(merged_df_2['birthday_5y'], q=10, precision=0, retbins=False, duplicates='drop')
merged_df_2['birthday_5y_binned'].value_counts()

# license_year
merged_df_2['license_year_5y_binned'] = pd.qcut(merged_df_2['license_year_5y'], q=5, precision=0, retbins=False, duplicates='drop')
merged_df_2['license_year_5y_binned'].value_counts()

# vhConstructionYear
merged_df_2['vhConstructionYear_binned'] = pd.qcut(merged_df_2['vhConstructionYear'], q=13, precision=0, retbins=False, duplicates='drop')
merged_df_2['vhConstructionYear_binned'].value_counts()

# NEWAgeOfVehicle
merged_df_2['NEWAgeOfVehicle_Binned'] = pd.qcut(merged_df_2['NEWAgeOfVehicle'], q=13, precision=0, retbins=False, duplicates='drop')
merged_df_2['NEWAgeOfVehicle_Binned'].value_counts()

# NEWAgeOfCustomer
merged_df_2['NEWAgeOfCustomer_Binned'] = pd.qcut(merged_df_2['NEWAgeOfCustomer'], q=10, precision=0, retbins=False, duplicates='drop')
merged_df_2['NEWAgeOfCustomer_Binned'].value_counts()

merged_df_2['NEWTotalLicenseYear_Binned'] = pd.qcut(merged_df_2['NEWTotalLicenseYear'], q=5, precision=0, retbins=False, duplicates='drop')
merged_df_2['NEWTotalLicenseYear_Binned'].value_counts()

dummy_cols_2 = ['NEWTotalLicenseYear_Binned','birthday_5y_binned','license_year_5y_binned','vhConstructionYear_binned','NEWAgeOfVehicle_Binned', 'NEWAgeOfCustomer_Binned']
merged_df_2 = dummy_encoding(merged_df_2,dummy_cols_2)

merged_df_2.columns = merged_df_2.columns.str.replace('[\[\]\(\)]', '', regex=True)
merged_df_2.head()

value_counts = merged_df_2['make'].value_counts()
print(value_counts/len(merged_df_2))
value_counts = merged_df_2['model'].value_counts()
print(value_counts/len(merged_df_2))

# Make and model creating Others group
threshold = 0.05
value_counts = merged_df_2['make'].value_counts()
total_records = len(merged_df_2)
less_frequent_values = value_counts[value_counts / total_records < threshold].index
merged_df_2['make'] = merged_df_2['make'].apply(lambda x: 'Others' if x in less_frequent_values else x)

merged_df_2 = dummy_encoding(merged_df_2,['make'])

# Model
(merged_df_2['model'].value_counts())/len(merged_df_2)

threshold = 0.01
value_counts = merged_df_2['model'].value_counts()
total_records = len(merged_df_2)
less_frequent_values = value_counts[value_counts / total_records < threshold].index
merged_df_2['model'] = merged_df_2['model'].apply(lambda x: 'Others' if x in less_frequent_values else x)

merged_df_2 = dummy_encoding(merged_df_2,['model'])

merged_df_2.head()

# WoE calculation for postal code:

def calculate_woe(train_idx, test_idx, feature, data, target):
    train = data.loc[train_idx]
    test = data.loc[test_idx]

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

    return train, test

for feature in ['postal_code_XX']:
    train_data, test_data = calculate_woe(
        train_indices, test_indices, feature, merged_df_2, 'converted'
    )

merged_df_2.head()

column_to_drop = ['license_year_5y','Year_Month','converted','dero','animation','animation_2',
                  'nbYearsAttest','profession', 'birthday_5y','fuelType',
                  'vhConstructionYear', 'mainDriverNotDesignated','purchaseTva', 'customerScore', 'model', 'make', 'postal_code_XX',
                  'nbbackoffice','premiumCustomer', 'NEWAgeOfVehicle',	'NEWAgeOfCustomer',	'NEWTotalLicenseYear',	'NEWAgeCustInterval',	'NEWAgeVehicleInterval',
                  'birthday_5y_binned',	'license_year_5y_binned','customerScore_Binned', 'vhConstructionYear_binned','NEWAgeOfVehicle_Binned',	'NEWAgeOfCustomer_Binned',	'NEWTotalLicenseYear_Binned']

merged_df_3 = merged_df_2.drop(column_to_drop, axis = 1)

X_train = merged_df_3.iloc[train_indices]
X_test = merged_df_3.iloc[test_indices]

Y_train = merged_df_2.iloc[train_indices]['converted']

"""
XGBOOST
"""

import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False)
best_params = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
best_xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, **best_params)
best_xgb_model.fit(X_train, np.ravel(Y_train))

# predictions on the test set with the best model
y_test_labels = best_xgb_model.predict(X_test)
y_test_scores = best_xgb_model.predict_proba(X_test)[:, 1]


# Creating a DataFrame with ID from 0 to 4999, y_test_labels, and y_test_scores
data = {
    'Column1': range(5000),  # Assuming 0 to 4999 as the ID column
    'converted': y_test_scores
}

df = pd.DataFrame(data)

# Exporting to Excel
df.to_excel('Prediction_Begum_Dilem_Linh.xlsx', index=False)  # Save to an Excel file without index

# Credit Risk Analysis

### Table of Contents:

# Stage 0: Problem Statement
---
## Overview
Determining the creditworthiness of a customer is crucial before lending a loan to avoid the high number of possible default customers. In order to assess the credit risk, we need to calculate three following metrics:
1. **Probability of Default**: is the likelihood that a borrower will fail to pay back a certain debt.
2. **Loss Given Default**: is the amount of money a financial institution loses when a borrower defaults on a loan, after taking into consideration any recovery, represented as a percentage of total exposure at the time of loss.
3. **Exposure at Default**: is the predicted amount of loss a bank may be exposed to when a debtor defaults on a loan.

## Objective
Develop a model to predict the **Expected Loss** by calculating the three main metrics mentioned above.

# Stage 1: Exploratory Data Analysis
---
## Checking Data Types
Using the ```.info()``` function allow us to understand the feature in a big picture. There are some features with **NULL** values in every record on it, thus it will be deleted in the future. There are also more features with **missing values** on it.<br>
Some features such as **emp_length, term**, and **date-value features** are also not in the correct types. Therefore, it will be handled in the next step.

# Stage 2: Data Preprocessing
---
## General Preprocessing
Firstly, the **emp_length** and the **term** will be converted into integer value and will be stored in the new feature called **emp_length_int** and **term_int**.
``` python
data['emp_length_int'] = data['emp_length'].str.replace('\+ years','')
data['emp_length_int'] = data['emp_length_int'].str.replace('< 1 year',str(0))
data['emp_length_int'] = data['emp_length_int'].str.replace('n/a',str(0))
data['emp_length_int'] = data['emp_length_int'].str.replace(' years','')
data['emp_length_int'] = data['emp_length_int'].str.replace(' year','')
data['emp_length_int'] = pd.to_numeric(data['emp_length_int'])
data.drop(columns=['emp_length'], inplace=True)

data['term_int'] = data['term'].str.replace(' months', '')
data['term_int'] = pd.to_numeric(data['term_int'])
data.drop(columns=['term'], inplace=True)
```

As for the date-value features like **earliest_cr_line, issue_d, last_pymnt_d, next_pymnt_d, and last_credit_pull_d** will be converted into date-time type.
```python
d_str = ['earliest_cr_line', 'issue_d', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']

data['earliest_cr_line_date'] = pd.to_datetime(data[d_str[0]], format='%b-%y')
data['issue_d_date'] = pd.to_datetime(data[d_str[1]], format='%b-%y')
data['last_pymnt_d_date'] = pd.to_datetime(data[d_str[2]], format='%b-%y')
data['next_pymnt_d_date'] = pd.to_datetime(data[d_str[3]], format='%b-%y')
data['last_credit_pull_d_date'] = pd.to_datetime(data[d_str[4]], format='%b-%y')
```

## Handling Missing Value
From all of the numeric features with the missing value, there are **four features** that will be imputed using **KNN Imputer** which is **'total_rev_hi_lim', 'annual_inc', 'revol_util', and 'tot_cur_bal'**. By using KNN Imputer, we can impute the missing value by using the prediction value based on the nearest neighbour. This way, we can have a more accurate values.<br>

As for the other, we will impute all the missing value with **zero (0)**
```python
from sklearn.impute import KNNImputer
nums_data = data[possible_knn_features].copy()

KNNImputer = KNNImputer(n_neighbors=5)
impute_nums_data = pd.DataFrame(KNNImputer.fit_transform(nums_data), columns=nums_data.columns)

for i in range(len(missing_nums_data)):
    data[missing_nums_data[i]].fillna(0, inplace=True)
```
Similarly, as for the missing categorical features, the imputation will be performed by using **Simple Imputer** method, where we will add another category called **missing** for all the missing value features.
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='constant', fill_value='missing')
impute_cats_data = pd.DataFrame(imputer.fit_transform(cats_data), columns = cats_data.columns)
```

## Feature Extraction
The feature extraction process will be focused on the date-time features, where we will calculate the day passed from the recorded date to the present. First, we extract the information from the **earliest_cr_line_date**
```python
time_passed_days = pd.to_datetime('2023-06-01') - data['earliest_cr_line_date']
data['mths_since_earliest_cr_line'] = round(pd.to_numeric(time_passed_days/np.timedelta64(1,'M')))
```
Since we have records of year 1960-1970 in the data frame, it will resulting the pandas to read as the year of 2060-2070 and returning a **negative value**, which doesn't make any sense. Therefore, it need to be replaced with more appropriate value.
```python
data['mths_since_earliest_cr_line'][data.mths_since_earliest_cr_line < 0] = data['mths_since_earliest_cr_line'].max()
```
The same goes with the date-time feautures.

# Stage 3: Modeling
---
>**_NOTE:_** To calculate the **Expected Loss**, the following formula is applied.<br>
> **EL = PD x LGD x EAD** <br>

## Probability of Default (PD)
### Splitting Data
The dependent feature for PD is called **good_bad**, which contain the binary information where **0=default/bad** and **1=non-default/good**. The **good_bad** value will be determined based on the **loan_status**.<br>
If the **loan_status**=**'Charged Off', 'Late (16-30 days)','Default'**, or **'Does not meet the credit policy. Status:Charged Off'** then the value is 0, otherwise, it's 1.
```python
data['good_bad'] = np.where(data['loan_status'].isin(['Charged Off', 'Late (16-30 days)','Default','Does not meet the credit policy. Status:Charged Off']),0,1)
```
The train and the test data will be divided with the proportion of **80:20**.
```python
X = data.drop(columns=['good_bad'])
y = data['good_bad']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
```

### Feature Selection - Weight of Evidence and Information Value
The use of **Weight of Evidence (WoE)** and **Information Value** have been used as a benchmark to screen variables in the credit risk modeling projects such as probability of default. WoE helps to understand if a particular class of an independent variable has a higher distribution of good or bad.

>**_WoE_** = **ln(distribution of goods / distribution of bads)**

Before we calculate the WoE, we need to know **the number of good** and **the number of bad** first.
```python
def woe_discreat(X_train, discreat_var_name, y_train):
    df = pd.concat([X_train[discreat_var_name], y_train], axis=1)
    
    #Calculate the number of observations and the proportion of good values (which is 1).
    #Since the good_bad feature only range from 0-1, calculate the mean good_bad will gives us the same result of the proportion of the good values (1).
    n_obs = df.groupby(discreat_var_name, as_index = False).agg(
        n_obs = (discreat_var_name, 'count')
    )
    prop_good = df.groupby(discreat_var_name, as_index = False).agg(
        prop_good  =(y_train.name,'mean')
    )
    
    df = pd.concat([n_obs, prop_good], axis=1)
    
    #Removing the duplicate feature (only return feature from index 0,1,and 3)
    df = df.iloc[:,[0,1,3]]
    
    #calculate the percentage of observations
    df['prop_n_obs'] = df['n_obs']/df['n_obs'].sum()
    
    #calculate the n_good and the n_bad using the following formula
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1-df['prop_good']) * df['n_obs']
    
    #calculate the percentage of the n_good and the n_bad
    df['prcnt_good'] = df['n_good']/df['n_good'].sum()
    df['prcnt_bad'] = df['n_bad']/df['n_bad'].sum()
    
    #calculate the WoE Score
    df['woe'] = np.log(df['prcnt_good'] / df['prcnt_bad'])
    
    #calculate Information Value (IV)
    df['IV'] = (df['prcnt_good'] - df['prcnt_bad'])*df['woe']
    df['IV'] = df['IV'].sum()
    
    #calculate the difference of good loan and the woe
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_woe'] = df['woe'].diff().abs()
    
    df = df.sort_values(by='woe')
    
    return df
```

Furthermore, to make the data more readable, we can visualize the plot of our WoE and IV calculation by using the following function.
```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_woe(df_woe, rotation_of_axis_labels=0):
    x = np.array(df_woe.iloc[:, 0].apply(str)) #retrieve the grade feature
    y = df_woe['woe']
    
    plt.figure(figsize=(18,6))
    plt.plot(x,y, marker = 'o', linestyle='--', color='k')
    plt.xlabel(df_woe.columns[0])
    plt.ylabel('Weight of Evidence')
    
    plt.title(str('Weight of Evidence by' + df_woe.columns[0]))
    plt.xticks(rotation = rotation_of_axis_labels)
    plt.grid()
```
### Feature Selection: Grade Feature
```python
woe_discreat(X_train, 'grade', y_train)
plot_woe((woe_discreat(X_train, 'grade', y_train)))
```
Running the codes above will resulting the following output.
![image](https://github.com/FluffyArc/eCommerce_Analysis/assets/40890491/97c0ea50-a0d6-47b5-b02d-6a2131ef09d4)
![image](https://github.com/FluffyArc/eCommerce_Analysis/assets/40890491/6e4ca289-f7f0-42f5-85a6-96ffb1b4ffa1)
>**_NOTE_**: **The negative** WoE value denotes that the **Distribution of Bad > Distribution of Good**. On the other hand, **positive** WoE value means **Distribution of Goods > Distribution of Bads**

Based on the grade's WoE plot, we can see that our bins has created a monotonic trend (ascending), therefore we can confirm that our bins have a general trend.<br>

As for the **Information Value**, it gives us the information about the **Predictive Power** that our feature have. Below table is showing the relation between IV and Predictive power of a variable.
![image](https://github.com/FluffyArc/eCommerce_Analysis/assets/40890491/5ce91052-cf2d-4dba-bfb0-141c0902f9c8)<br>
*image by Anik Chakraborty (medium.com)*

Any variable having IV **lesser than .02** can be **excluded** in our binary logistic regression model. As for our *grade* feature, it generate an IV score of **0.277**. Therefore, we can confirm that the *grade* feature is a **medium predictor**
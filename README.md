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
### Feature Selection: Categorical Features
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

As for the **Information Value**, it gives us the information about the **Predictive Power** that our feature have. Below table is showing the relation between IV and Predictive power of a variable.<br>
![image](https://github.com/FluffyArc/eCommerce_Analysis/assets/40890491/5ce91052-cf2d-4dba-bfb0-141c0902f9c8)<br>
*image by Anik Chakraborty (medium.com)*

Any variable having IV **lesser than .02** can be **excluded** in our binary logistic regression model. As for our *grade* feature, it generate an IV score of **0.277**. Therefore, we can confirm that the *grade* feature is a **medium predictor**

### Feature Selection: Home_Ownership
Running the function ```woe_discreat``` and ```plot_woe``` will generate the following output.<br>
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/a5a9c3c1-6580-47a3-8045-6a838ebc0d75)

As we can see, since the **n_obs (number of observations)** for **OTHER, NONE, ANY,** and **OWN** are relatively small compared to the rest, we can merge those categories into a new one called **home_ownership:RENT_OTHER_NONE_ANY**.

```python
X_train['home_ownership:RENT_OTHER_NONE_ANY'] = sum([
    X_train['ownership:RENT'], X_train['ownership:OTHER'],
    X_train['ownership:NONE'], X_train['ownership:ANY']
])
```

The Information Value (IV) for **home_ownership** feature is **inf(infinity)**, which means this feature an **extremely powerful predictor**

### Feature Selection: Addr_State
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/ddfabae4-c07f-4ac1-b808-7d0f7cbb125c)

As for the **addr_state** feature, since it has 50 different value, extracting some new features like in the previous step also neccessary. By using ```plot_woe(woe_discreat(X_train, 'addr_state',y_train).iloc[8:-2, :])``` command, we can have a clear picture of the **addr_state** plot.<br>
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/3838b504-4e5e-4d26-b9c2-9902f88c02ee)

Once again, the extraction of the new features can be determined based on their **n_obs**. The Information Value (IV) for **addr_state** feature is **inf(infinity)**, which means this feature an **extremely powerful predictor**

### Feature Selection: Purpose
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/e641720e-d0ea-44b9-a6d3-0555fdfa0649)

The Information Value (IV) for **purpose** feature is **0.041**, which means this feature is a **weak predictor**.

### Feature Selection: Verification Status and Initial List Status
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/59a18b3e-14e1-4a32-a5da-f688767f3eac)
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/f0d9bf8c-2885-4f18-b278-058c4a9257fe)

The Information Value (IV) for both **verification_status** and **initial_list_status** feature are **0.022** and **0.040** consecutively, which means those features are **weak predictors**.

### Feature Selection: Numerical Features
Not only the categorical features, we also need to encode the numerical features in order to train our model more accurate. The feature selection method for numerical features is the same like the categorical features. The following **WoE Calculation** is used to calculate the WoE and the IV of the features.
```python
def woe_ordered_continuous(X_train, cont_var_name, y_train):
    df = pd.concat([X_train[cont_var_name], y_train], axis=1)
    
    #Calculate the number of observations and the proportion of good values (which is 1)
    #Since the good_bad feature only range from 0-1, calculate the mean good_bad will gives us the same result of the proportion of the good values (1)
    n_obs = df.groupby(cont_var_name, as_index = False).agg(n_obs = (cont_var_name, 'count'))
    prop_good = df.groupby(cont_var_name, as_index = False).agg(prop_good = (y_train.name,'mean'))
    
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
    
    return df
```

### Feature Selection: Term
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/6d1d96e4-6dd2-4afd-8db8-87ab86336a1e)

The Information Value (IV) for **term_int** feature is **0.034**, which means this features is a **weak predictors**

### Feature Selection: Employment Length
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/0f9cd13d-2a98-4677-83e1-5fdf96bdf61d)

The Information Value (IV) for **emp_length_int** feature is **0.007**, which means this features is **not useful** and we can exclude this feature as our predictor.

### Feature Selection: Months Since Issue Date
The **mths_since_issue_date** feature returning a value with vary categories. Therefore, we need to extract a new category that could cover all the values. ```pd.cut()``` can generate the desired categories based on the limit setting. <br>
```python
#We tried to divide the value into 50 different category
pd.cut(X_train['mths_since_issue_date'], 50)
```
Once again, the extraction of the new features can be determined based on their **n_obs**.<br>
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/cf6bfec3-f1e7-40bf-ab40-da794954a2c5)

The Information Value (IV) for **mths_since_issue_date** feature is **0.016**, which means this features is a **weak predictor**.

### Feature Selection: Int Rate
Similar to the previous feature, since **int_rate** returned a vary values, we need to extract a new category from it by using ```pd.cut```.<br>

![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/32f60402-8603-46f2-87f8-aa373bea24f6)

The Information Value (IV) for **int_rate** feature is **0.345**, which means this features is a **strong predictor**.

### Feature Selection: Funded Amount
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/2d83bd17-ba45-4b06-b88f-66be83364931)

The Information Value (IV) for **funded_amnt** feature is **0.014**, which means this features is **not useful** and we can exclude this feature as our predictor.

### Feature Selection: Months Since Earliest Credit Line
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/f742bc29-6a88-409f-9149-94626509bc34)

The Information Value (IV) for **mths_since_earliest_cr_line** feature is **0.016**, which means this features is **not useful** and we can exclude this feature as our predictor.

### Feature Selection: Installment
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/9f4d21c4-edc7-436c-b431-758467b0c2b0)

The Information Value (IV) for **installment** feature is **0.013**, which means this features is **not useful** and we can exclude this feature as our predictor.

### Feature Selection: Delinquent for 2 Years
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/1f1b6940-df4e-40b6-80ca-04c6e88d058a)

The Information Value (IV) for **delinq_2yrs** feature is **inf(infinity)**, which means this feature an **extremely powerful predictor**.

### Feature Selection: Inqueries in Last 6 Months
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/9747a488-fc7b-43a6-bfcf-4cf17d3dd448)

The Information Value (IV) for **inq_last_6mths** feature is **inf(infinity)**, which means this feature an **extremely powerful predictor**.

### Feature Selection: Open Account
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/39ce4986-f1f5-4d5c-ac14-d615fc8f52a1)

The Information Value (IV) for **open_acc** feature is **inf(infinity)**, which means this feature an **extremely powerful predictor**.

### Feature Selection: Public Record
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/6b272845-ff31-4d36-9620-8e8b39c853d7)
The Information Value (IV) for **pub_rec** feature is **inf(infinity)**, which means this feature an **extremely powerful predictor**.

### Feature Selection: Account Now Delinquent
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/e2b6d5d7-4d1c-4b2b-8688-e8fe50437f78)
The Information Value (IV) for **acc_now_delinq** feature is **inf(infinity)**, which means this feature an **extremely powerful predictor**.

### Feature Selection: Annual Income
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/9fe21ade-8dac-486d-a598-29d16d37e937)
The Information Value (IV) for **annual_inc** feature is **0.05365**, which means this features is a **weak predictor**.

### Feature Selection: Months Since Last Delinquent
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/e15a32d2-4c64-4382-a61a-d82136161626)
The Information Value (IV) for **mths_since_last_delinq** feature is **inf(infinity)**, which means this feature an **extremely powerful predictor**.

### Feature Selection: DTI
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/f562fe16-15b3-4c92-b82f-f671ab045394)
The Information Value (IV) for **dti** feature is **0.023004**, which means this feature is a **weak predictor**.

### Feature Selection: Months Since Last Record
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/4d2aaf6d-5be7-407c-aac8-02ac5341f6f4)
The Information Value (IV) for **mths_since_last_record** feature is **0.010336**, which means this features is **not useful** and we can exclude this feature as our predictor.

## Feature Extraction: Numerical Features.
To create a more readable feature, a new group of selected numerical features will be extracted from the existing features. The new group will be created based on the **num_obs** by using the ```plot_woe()``` function.

# Model Fitting
**Logistic Regression** is one of the simplest models that can be used to predict the Probability of Default. 

# Model Evaluation
## ROC_AUC Curve
By using the ROC Curve, we can identify how good our model's performance is.
![image](https://github.com/FluffyArc/CreditRisk/assets/40890491/518511f1-206c-4a07-b1ac-a0a0b924fb96)

By calculating the auc_roc from the real data test and the predicted probability, we get the score of **0.683**. A binary classifier is **useful only** when it achieves ROC-AUC score **greater than 0.5 and as near to 1 as possible**. If a classifier yields a score **less than 0.5**, it simply means that the **model is performing worse** than a random classifier, and hence, is of no use.

## Gini Coefficient
The Gini coefficient is a metric that indicates the model’s discriminatory power, namely, the effectiveness of the model in differentiating between **bad** borrowers, who will default in the future, and **good** borrowers, who won’t default in the future.<br>

Gini score can be calculated with the following formula:<br>
```gini = auc_roc*2-1```. The model get **0.3672** for the Gini Coefficient score. Even though it's still far from the perfect model, our trained model has the power to distinguish the good and bad borrower.
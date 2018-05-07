
# coding: utf-8

# Machine learning on loan data set

import hbase
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category = matplotlib.cbook.mplDeprecation)
import numpy as np
import pandas as pd


hbase.init() 

The most common function for loading data into an hbase cluster is to use the [`import_file`](http://docs.hbase.ai/hbase/latest-stable/hbase-py/docs/hbase.html?highlight=import_file#hbase.import_file) on files that are visible to the cluster using either a server-side file system path, HDFS path, S3, or a URL. If the data files are local to the Python client, the [`upload_file`](http://docs.hbase.ai/hbase/latest-stable/hbase-py/docs/hbase.html?highlight=upload_file#hbase.upload_file) function uploads files based upon a local path to the hbase cluster.

train_path = "C:/Users/KNOZE/Documents/MLPROJECTS/loan.csv"

train = hbase.import_file(train_path, destination_frame = "loan_train")
type(train)
train.col_names
train.types
print(train.frame_id)
train["bad_loan"] = train["bad_loan"].asfactor()
print(train.frame_id)


# Exploratory Data Analysis

# Look at the Target Variable
mytable = train["bad_loan"].table()
mytable = mytable.as_data_frame()

print(mytable)
print("\nFraction of bad_loan = {:0.4f}".format(mytable[mytable["bad_loan"] == 1]["Count"].values[0] / tbl["Count"].sum()))

# Features of the data set 

train.describe()
{ train.col_names[int(j)] : train[int(j)].skewness(na_rm = True)[0] for j in train.columns_by_type("numeric") }
{ train.col_names[int(j)] : train[int(j)].kurtosis(na_rm = True)[0] for j in train.columns_by_type("numeric") }

for Q in train.columns_by_type("numeric"):
    train[int(Q)].hist()
plt.show()


#Feature Engineering in hbase

# Since the goal of this analysis is to predict credit card default, we will create a string, `y`, for the response variable, "bad_loan", and a list of column names for the original set of predictors, `x_orig`. The original set of predictors should not include interest rate since they were set based upon a risk assessment that the loan would be bad.
# Improve upon the predictors through a number of feature engineering steps. 
y = "bad_loan"
x_myoriginal = train.col_names
x_myoriginal.remove(y)
x_myoriginal.remove("int_rate")
x_trans = x_myoriginal.copy()

# Cross Validation and Target Encoding
cv_nfolds = 10
cv_seed = 1234567
train["cv_fold"] = train.kfold_column(n_folds = cv_nfolds, seed = cv_seed)
train["cv_fold"].table()
def logit(p):
    return np.log(p) - np.log(1 - p)
def mean_target(data, x, y = "bad_loan"):
    grouped_data = data[[x, y]].group_by([x])
    stats = grouped_data.count(na = "ignore").mean(na = "ignore")
    return stats.get_frame().as_data_frame()

def mean_target_encoding(data, x, y = "bad_loan", fold_column = "cv_fold", prior_mean = 0.183, prior_count = 1):
    """
    Creates target encoding for binary target
    data (hbaseFrame) : data set
    x (string) : categorical predictor column name
    y (string) : binary target column name
    fold_column (string) : cross-validation fold column name
    prior_mean (float) : proportion of 1s in the target column
    prior_count (positive number) : weight to give to prior_mean
    """ 
    grouped_data = data[[x, fold_column, y]].group_by([x, fold_column])
    grouped_data.sum(na = "ignore").count(na = "ignore")
    df = grouped_data.get_frame().as_data_frame()
    df_list = []
    nfold = int(data[fold_column].max()) + 1
    for j in range(0, nfold):
        te_x = "te_{}".format(x)
        sum_y = "sum_{}".format(y)
        oof = df.loc[df[fold_column] != j, [x, sum_y, "nrow"]]
        stats = oof.groupby([x]).sum()
        stats[x] = stats.index
        stats[fold_column] = j
        p = (stats[sum_y] + (prior_count * prior_mean)) / (stats["nrow"] + prior_count)
        stats[te_x] = logit(p)
        df_list.append(stats[[x, fold_column, te_x]])
    return hbase.hbaseFrame(pd.concat(df_list))

train["loan_amnt"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
breaks = np.linspace(0, 35000, 8).tolist()
train["loan_amnt_cat"] = train["loan_amnt"].cut(breaks = breaks)
df = mean_target(train, "loan_amnt_cat")
plt.xticks(rotation = 90)
plt.yscale("logit")
plt.plot(df["loan_amnt_cat"], df["mean_bad_loan"])
df
x_trans.remove("loan_amnt")
x_trans.append("loan_amnt_core")
x_trans.append("loan_amnt_delta")

train["loan_amnt_core"] = hbase.hbaseFrame.ifelse(train["loan_amnt"] <= 5000, 5000, train["loan_amnt"])
train["loan_amnt_core"] = hbase.hbaseFrame.ifelse(train["loan_amnt_core"] <= 30000, train["loan_amnt_core"], 30000)

train["loan_amnt_delta"] = train["loan_amnt"] - train["loan_amnt_core"]


# Convert Term to a 0/1 Indicator

# Given that term of the loans are either 3 or 5 years, we will create a simplifed `term_36month` binary indicator that is 1 when the terms of the loan is for 5 years and 0 for loans with a term of 3 years.
train["term"].table()
x_trans.remove("term")
x_trans.append("term_60months")
train["term_60months"] = train["term"] == "60 months"
train["term_60months"].table()
#Creating Missing Value Indicator for Employment Length
train["emp_length"].summary()
x_trans.append("emp_length_missing")
train["emp_length_missing"] = train["emp_length"] == None
mean_target_encoding(train, "emp_length_missing")
df = mean_target(train, "emp_length")
plt.yscale("logit")
plt.plot(df["emp_length"], df["mean_bad_loan"])
df
#  Combining Categories in Home Ownership

# Although there are 6 recorded categories within home ownership, only three had over 200 observations: OWN, MORTGAGE, and RENT. The remaining three are so infrequent we will combine them {ANY, NONE, OTHER} with RENT to form an enlarged OTHER category. This new `home_ownership_3cat` variable will have values in {MORTGAGE, OTHER, OWN}.
mean_target(train, "home_ownership")

lvls = ["OTHER", "MORTGAGE", "OTHER", "OTHER", "OWN", "OTHER"]
train["home_ownership_3cat"] = train["home_ownership"].set_levels(lvls).ascharacter().asfactor()
train[["home_ownership", "home_ownership_3cat"]].table()
mean_target(train, "home_ownership_3cat")
x_trans.remove("home_ownership")
x_trans.append("home_ownership_3cat")
train["annual_inc"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
breaks = np.linspace(0, 150000, 31).tolist()
train["annual_inc_cat"] = train["annual_inc"].cut(breaks = breaks)
df = mean_target(train, "annual_inc_cat")
plt.yscale("logit")
plt.plot(df["annual_inc_cat"].index, df["mean_bad_loan"])
df[0:6]
df[20:31]
x_trans.remove("annual_inc")
x_trans.append("annual_inc_core")
x_trans.append("annual_inc_delta")
train["annual_inc_core"] = hbase.hbaseFrame.ifelse(train["annual_inc"] <= 10000, 10000, train["annual_inc"])
train["annual_inc_core"] = hbase.hbaseFrame.ifelse(train["annual_inc_core"] <= 105000,
                                               train["annual_inc_core"], 105000)
train["annual_inc_delta"] = train["annual_inc"] - train["annual_inc_core"]

# Creating Target Encoding for Loan Purpose

# Given that there is a high concentration of loans for debt consolidation (56.87%), a sizable number for credit card (18.78%), and the remaining 24.35% loans are spread amongst 12 other purposes, we will use mean target encoding to avoid overfitting of the later group.
tbl = train["purpose"].table().as_data_frame()
tbl["Percent"] = np.round((100 * tbl["Count"]/train.nrows), 2)
tbl = tbl.sort_values(by = "Count", ascending = 0)
tbl = tbl.reset_index(drop = True)
print(tbl)
mean_target(train, "purpose")
te_purpose = mean_target_encoding(train, "purpose")
train = train.merge(te_purpose, all_x = True)
x_trans.remove("purpose")
x_trans.append("te_purpose")
train["te_purpose"].hist()

# #### Creating Target Encoding for State of Residence
# We will also use a mean target encoding for state of residence for a reason similar to that for purpose.
tbl = train["addr_state"].table().as_data_frame()
tbl["Percent"] = np.round((100 * tbl["Count"]/train.nrows), 2)
tbl = tbl.sort_values(by = "Count", ascending = 0)
tbl = tbl.reset_index(drop = True)
print(tbl[0:5])
df = mean_target(train, "addr_state")
plt.yscale("logit")
plt.plot(df["addr_state"], df["mean_bad_loan"])
te_addr_state = mean_target_encoding(train, "addr_state", prior_count = 30)
train = train.merge(te_addr_state, all_x = True)
x_trans.remove("addr_state")
x_trans.append("te_addr_state")
train["te_addr_state"].hist()

# #### Separating Typical from Extreme Debt to Income Ratio
train["dti"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
breaks = np.linspace(0, 40, 41).tolist()
train["dti_cat"] = train["dti"].cut(breaks = breaks)
df = mean_target(train, "dti_cat")
plt.yscale("logit")
plt.plot(df["dti_cat"].index, df["mean_bad_loan"])
df[30:41]
x_trans.remove("dti")
x_trans.append("dti_core")
x_trans.append("dti_delta")

train["dti_core"] = hbase.hbaseFrame.ifelse(train["dti"] <= 5, 5, train["dti"])
train["dti_core"] = hbase.hbaseFrame.ifelse(train["dti_core"] <= 30, train["dti_core"], 30)

train["dti_delta"] = train["dti"] - train["dti_core"]

# Separating Typical from Extreme Number of Delinquencies in the Past 2 Years

train["delinq_2yrs"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
breaks = np.linspace(0, 5, 6).tolist()
train["delinq_2yrs_cat"] = train["delinq_2yrs"].cut(breaks = breaks)
mean_target(train, "delinq_2yrs_cat")
x_trans.remove("delinq_2yrs")
x_trans.append("delinq_2yrs_core")
x_trans.append("delinq_2yrs_delta")
train["delinq_2yrs_core"] = hbase.hbaseFrame.ifelse(train["delinq_2yrs"] <= 3, train["delinq_2yrs"], 3)
train["delinq_2yrs_delta"] = train["delinq_2yrs"] - train["delinq_2yrs_core"]
# Separating Typical from Extreme Revolving Credit Line Utilized
# The relationship between credit line utilized is somewhat interesting. There appears to be a higher rate for a bad loan when 0% of the credit lines are utilized, then it drops down slightly and roughly increases linearly in credit line utilized up to 100%. To reflect this finding in the modeling, we will replace the original `revol_util` measure with three derived measures:
train["revol_util"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
breaks = np.linspace(0, 120, 25).tolist()
train["revol_util_cat"] = train["revol_util"].cut(breaks = breaks)
df = mean_target(train, "revol_util_cat")
plt.yscale("logit")
plt.plot(df["revol_util_cat"].index, df["mean_bad_loan"])
df[20:25]
x_trans.remove("revol_util")
x_trans.append("revol_util_0")
x_trans.append("revol_util_core")
x_trans.append("revol_util_delta")

train["revol_util_0"] = train["revol_util"] == 0

train["revol_util_core"] = hbase.hbaseFrame.ifelse(train["revol_util"] <= 100, train["revol_util"], 100)

train["revol_util_delta"] = train["revol_util"] - train["revol_util_core"]

# #### Separating Typical from Extreme Number of Credit Lines

train["total_acc"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
breaks = np.linspace(0, 60, 13).tolist()
train["total_acc_cat"] = train["total_acc"].cut(breaks = breaks)
df = mean_target(train, "total_acc_cat")
plt.yscale("logit")
plt.plot(df["total_acc_cat"].index, df["mean_bad_loan"])
(train["total_acc"] == None).table()
df[0:3]
df[8:13]
x_trans.remove("total_acc")
x_trans.append("total_acc_core")
x_trans.append("total_acc_delta")

train["total_acc_core"] = hbase.hbaseFrame.ifelse(train["total_acc"] <= 50, train["total_acc"], 50)

train["total_acc_delta"] = train["total_acc"] - train["total_acc_core"]

# Longest Credit Length
train["longest_credit_length"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
breaks = np.linspace(0, 25, 26).tolist()
train["longest_credit_length_cat"] = train["longest_credit_length"].cut(breaks = breaks)
df = mean_target(train, "longest_credit_length_cat")
plt.yscale("logit")
plt.plot(df["longest_credit_length_cat"].index, df["mean_bad_loan"])
df[20:26]

x_trans.remove("longest_credit_length")
x_trans.append("longest_credit_length_core")
x_trans.append("longest_credit_length_delta")

train["longest_credit_length_core"] = hbase.hbaseFrame.ifelse(train["longest_credit_length"] <= 3,
                                                          3, train["longest_credit_length"])
train["longest_credit_length_core"] = hbase.hbaseFrame.ifelse(train["longest_credit_length_core"] <= 20,
                                                          train["longest_credit_length_core"], 20)

train["longest_credit_length_delta"] = train["longest_credit_length"] - train["longest_credit_length_core"]

# Converting Income Verification Status to a 0/1 Indicator

train["verification_status"].table()
x_trans.remove("verification_status")
x_trans.append("verified")

train["verified"] = train["verification_status"] == "verified"


# In[67]:


train["verified"].table()


# Saving the Transformed Data to a Known Key

print(train.frame_id)

hbase.assign(train, "loan_transformed")
train = hbase.get_frame("loan_transformed")
print(train.frame_id)

# Supervised Learning in hbase

print("Response = " + y)
print("Predictors (Orig) = " + str(x_orig))
print("Predictors (Trans) = " + str(x_trans))
from hbase.estimators.glm import hbaseGeneralizedLinearEstimator
help(hbaseGeneralizedLinearEstimator)
glm_orig_0 = hbaseGeneralizedLinearEstimator(family = "binomial", lambda_search = True,
                                           nfolds = cv_nfolds, fold_column = "cv_fold")
glm_trans_0 = hbaseGeneralizedLinearEstimator(family = "binomial", lambda_search = True,
                                            nfolds = cv_nfolds, fold_column = "cv_fold")
glm_orig_0.train(x = x_myoriginal, y = y, training_frame = train, model_id = "bad_loan_glm_orig_0")
glm_trans_0.train(x = x_trans, y = y, training_frame = train, model_id = "bad_loan_glm_trans_0")
print("Log Loss_GLM(orig)  = {:0.4f}".format(glm_orig_0.logloss(xval = True)))
print("Log Loss_GLM(trans) = {:0.4f}".format(glm_trans_0.logloss(xval = True)))
print("AUC_GLM(orig)  = {:0.4f}".format(glm_orig_0.auc(xval = True)))
print("AUC_GLM(trans) = {:0.4f}".format(glm_trans_0.auc(xval = True)))


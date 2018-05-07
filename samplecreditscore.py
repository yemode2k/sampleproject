
# coding: utf-8

# Import necessary packages
import Algo
from Algo.estimators.gbm import AlgoGradientBoostingEstimator
from Algo.estimators.glm import AlgoGeneralizedLinearEstimator
# Initialize instance of Algo
Algo.init()
# If possible download from the s3 link and change the path to the dataset
path = "C:/Users/KNOZE/Documents/MLPROJECTS/loan.csv"
# Specify some column types to "String" that we want to munge later
types = {"int_rate":"string", "revol_util":"string", "emp_length":"string", "verification_status":"string"}

# FIle import
loan_stats = Algo.import_file(path=path, col_types= types)
loan_stats.describe()

# The response column, "loan_status"
# Hint: Use .table() function on the response column
loan_stats["loan_status"].table()

# TRemove on going loans
toremove = ["Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"]
loan_stats = loan_stats[loan_stats["loan_status"].isin(toremove).logical_negation(), :]
loan_stats.show()

# Bin the response variable to good/bad oans
# Create a new column called "bad_loan" which should be a binary variable
# Hint: You can turn the bad_loan columm into a factor using .asfactor()
loan_stats["bad_loan"] = (loan_stats["loan_status"] == "Fully Paid").logical_negation()
loan_stats["bad_loan"] = loan_stats["bad_loan"].asfactor()

# String munging removing  "int_rate," "revol_util," "emp_length"
loan_stats["int_rate"] = loan_stats["int_rate"].gsub(pattern = "%", replacement = "") # strip %
loan_stats["int_rate"] = loan_stats["int_rate"].trim() # trim ws
loan_stats["int_rate"] = loan_stats["int_rate"].asnumeric() #change to a numeric 
loan_stats["int_rate"].show()

# Now try for revol_util yourself
loan_stats["revol_util"] = loan_stats["revol_util"].gsub(pattern = "%", replacement = "")
loan_stats["revol_util"] = loan_stats["revol_util"].trim() 
loan_stats["revol_util"] = loan_stats["revol_util"].asnumeric() 
loan_stats["revol_util"].show()

# Now we're going to clean up emp_length
# Use gsub to remove " year" and " years" also translate n/a to "" 
loan_stats["emp_length"] = loan_stats["emp_length"].gsub(pattern = "([ ]*+[a-zA-Z].*)|(n/a)", 
                                              replacement = "") 

# Use trim to remove any trailing spaces 
loan_stats["emp_length"] = loan_stats["emp_length"].trim()

# Use sub to convert < 1 to 0 years and do the same for 10 + to 10
# Hint: Be mindful of spaces between characters
loan_stats["emp_length"] = loan_stats["emp_length"].gsub(pattern = "< 1", replacement = "0")
loan_stats["emp_length"] = loan_stats["emp_length"].gsub(pattern = "10\\+", replacement = "10")
loan_stats["emp_length"] = loan_stats["emp_length"].asnumeric()
loan_stats["emp_length"].show()

# Create new column called credit_length
# Hint: Do this by subtracting the earliest_cr year from the issue_d year
loan_stats["credit_length"] = loan_stats["issue_d"].year() - loan_stats["earliest_cr_line"].year()
loan_stats["credit_length"].show()

#  Use the sub function to create two levels from the verification_status column. Ie "verified" and "not verified"
loan_stats["verification_status"] = loan_stats["verification_status"].sub(pattern = "VERIFIED - income source", 
                                                               replacement = "verified")
loan_stats["verification_status"] = loan_stats["verification_status"].sub(pattern = "VERIFIED - income", 
                                                               replacement = "verified")
loan_stats["verification_status"] = loan_stats["verification_status"].asfactor()

# TTest-train split (80-20)
splits = loan_stats.split_frame(ratios = [0.80])
train = splits[0]
test  = splits[1]

# Response and predictor variables 
y="bad_loan"
x=["loan_amnt", "term", "home_ownership", "annual_inc", "verification_status", "purpose",
   "addr_state", "dti", "delinq_2yrs", "open_acc", "pub_rec", "revol_bal", "total_acc",
   "emp_length", "credit_length", "inq_last_6mths", "revol_util"]

# Train GBM Model
# Set parameters for GBM model 
from Algo.estimators.gbm import AlgoGradientBoostingEstimator
gbm_model = AlgoGradientBoostingEstimator(model_id="GBM_BadLoan",
                                         score_each_iteration=True,
                                         ntrees=100,
                                         learn_rate=0.05)


# Model Building
gbm_model.train(x=x, y=y, training_frame=train, validation_frame=test)


# scoring history to make sure you're not overfitting
# Hint: Use plot function on the model object
get_ipython().run_line_magic('matplotlib', 'inline')
gbm_model.plot()

#  Plot the ROC curve for the binomial models and get auc using Algo.auc
# Hint: Use Algo.performance and plot to grab the modelmetrics and then plotting the modelmetrics

gbm_model.model_performance(train = True).plot()
gbm_model.model_performance(valid = True).plot()

print "Training AUC = " + str(gbm_model.auc(train = True))
print "Validation AUC = " + str(gbm_model.auc(valid = True))

# Check the variable importance and generate confusion matrix for max F1 threshold

print gbm_model.varimp(use_pandas = True)
print gbm_model.confusion_matrix(valid = True)

#  Score the entire data set using the model
pred = gbm_model.predict(loan_stats)

# Extra: Calculate the money gain/loss if model is implemented
# Calculate the total amount of money earned or lost per loan
loan_stats["earned"] = loan_stats["total_pymnt"] - loan_stats["loan_amnt"]

# Calculate how much money will be lost to false negative, vs how much will be saved due to true positives
loan_stats["pred"] = pred["predict"]

grouped = loan_stats.group_by(["bad_loan", "pred"])
net = grouped.sum(col = "earned", na = "ignore").get_frame()

n1 = net[(net["bad_loan"] == "0") & (net["pred"] == "0")]["sum_earned"].round(digits = 0).max()
n2 = net[(net["bad_loan"] == "0") & (net["pred"] == "1")]["sum_earned"].round(digits = 0).max()
n3 = net[(net["bad_loan"] == "1") & (net["pred"] == "1")]["sum_earned"].round(digits = 0).max()
n4 = net[(net["bad_loan"] == "1") & (net["pred"] == "0")]["sum_earned"].round(digits = 0).max()

# Calculate the amount earned
print "Total amount of profit still earned using the model : %s" % '${:0,.0f}'.format(n1)
print "Total amount of profit forfeitted using the model : %s" % '${:0,.0f}'.format(n2)
print "Total amount of loss that could have been prevented : %s" % '${:0,.0f}'.format(n3)
print "Total amount of loss that still would've accrued : %s" % '${:0,.0f}'.format(n4)

# Calculate Net
print "Total profit by implementing model : $ %s" %'${:0,.0f}'.format((n1 - n2 + (-1*n3) - (-1*n4)))


# In[26]:


# Shutdown Algo instance
Algo.cluster().shutdown()



# Starting a local Algo cluster
library(Halgo)
Halgo.init(nthreads = -1) # -1 means using all cores


# ------------------------------------------------------------------------------
 Importing Data 
# ------------------------------------------------------------------------------

# Import CSV directly 
coverdata <- Halgo.importFile("")

# Turn "specialness" into categorial variable
coverdata[, "specialness"] <- as.factor(coverdata[, "specialness"])

# Have a quick look at data
head(coverdata)
summary(coverdata)


# ------------------------------------------------------------------------------
Splitting Data into Training / Validation / Test
# ------------------------------------------------------------------------------

# Split 
split_hex <- Halgo.splitFrame(coverdata, ratios = c(0.8, 0.15), seed = 1234)

# Creating three new HDFRAMEs

training_frame <- split_hex[[1]]
validation_frame <- split_hex[[2]]
test_frame <- split_hex[[3]]


# ------------------------------------------------------------------------------
# Building a Gradient Boosting Machines (GBM) Model
# ------------------------------------------------------------------------------

# Define features (predictors) and response
response <- "specialness"
features <- setdiff(colnames(training_frame), response)
print(features)
print(response)

# Training GBM with default values
model_gbm <- Halgo.gbm(x = features,
                     y = response,
                     training_frame = training_frame,
                     validation_frame = validation_frame)

# Looking at the model
print(model_gbm)

# Look at the model in details
print(summary(model_gbm))


# ------------------------------------------------------------------------------
 Model Predictions
# ------------------------------------------------------------------------------

# Predicting the "specialness" column in test_frame
yhat_test <- Halgo.predict(model_gbm, newdata = test_frame)

# the results
print(yhat_test)
Halgo.performance(model_gbm, newdata = test_frame)

# Converting the HDFRAME into normal R data frame for other analysis
yhat_test_df <- as.data.frame(yhat_test)



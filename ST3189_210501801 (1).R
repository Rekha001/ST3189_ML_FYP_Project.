# ===================================================================================================
# Topic:        Machine Learning using Unsupervised learning, Classification and Regression 
# Student ID:   210501801
# DOC:          03-04-2024
# Data Source:  Pima Indian diabetes dataset.csv
#====================================================================================================

# ~Pima Indian dataset description~ 

# The non-insulin-dependent diabetes mellitus is a disease commonly found in pima indians. It is hereditary and strongly related to obesity. 
# This dataset contains 768 observations of pima Indian women between ages 21 to 81 with 8 input variables and 1 output variable.
# The list below represents the various attributes of the pima Indian dataset.
#---
# Pregnancies:              Number of times pregnant
# Glucose:                  Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# BloodPressure:            Diastolic blood pressure (mm Hg)
# SkinThickness:            Triceps skin fold thickness (mm)
# Insulin:                  2-Hour serum insulin (mu U/ml)
# BMI:                      Body mass index (weight in kg/(height in m)^2)
# DiabetesPedigreeFunction: Diabetes pedigree function (a function that represents how likely they are to get the disease by extrapolating from their ancestors)
# Age:                      Age (years)
# Outcome:                  1: Tested positive for diabetes, 0: Tested negative for diabetes 
#---

# installing relevant packages to be used in this dataset
library(rpart)
library(rpart.plot)
library(nnet)
library(dplyr)
library(zoo)
library(neuralnet)
library(caTools)
library(tidyverse)
library(ggcorrplot)
library(class)
library(caret)
library(ggplot2)
library(pROC)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)

# Loading and analysing data
diabetes_data <- read.csv("diabetes.csv")
colSums(is.na(diabetes_data)) # There are no missing data
head(diabetes_data, n=10)     
str(diabetes_data)
summary(diabetes_data)
# we can see that the minimum value for Glucose, BloodPressure, SkinThickness, Insulin and BMI is 0. 
# It does not make sense for those values to be 0. 


# EDA 
diabetes_data_copy <- diabetes_data
# We define a vector columns_to_replace containing the names of the columns we want to replace the 0 values with NA.
columns_to_replace <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
# We use the lapply() function to iterate over the specified columns and replace the 0 values with NA.
diabetes_data_copy[columns_to_replace] <- lapply(diabetes_data_copy[columns_to_replace], function(x) {ifelse(x == 0, NA, x)})
colSums(is.na(diabetes_data_copy))
# With a total of 768 values, NA values of each x variable is shown below.

# Insulin       = 374
# SkinThickness = 227
# BloodPressure = 35
# BMI           = 11 
# Glucose       = 5 


# Calculate the percentage of missing values for each column
missing_values <- colSums(is.na(diabetes_data_copy))
total_observations <- nrow(diabetes_data_copy)
percentage_missing <- (missing_values / total_observations) * 100
print(percentage_missing)

# Almost 50% of insulin values are missing while 30% of SkinThickness are missing. 
# *With a small dataset coupled with a large number of NA values, dropping those NA values will drastically affect our results.*
# To tackle this situation, we will have to understand the data distribution. 

# Setting up the plotting area of 1x1 grid 
par(mfrow=c(1, 1))  

# Iterating through each column and plotting histograms
for (col in names(diabetes_data_copy)) {
  hist(diabetes_data_copy[[col]], 
       main = paste("Histogram of", col),
       xlab = col,
       ylab = "Frequency",
       col = "blue",
       border = "black")
}

# Replace NA values in accordance to their distributions.  
# Replace missing values with mean for 'Glucose' column
diabetes_data_copy$Glucose <- ifelse(is.na(diabetes_data_copy$Glucose), mean(diabetes_data_copy$Glucose, na.rm = TRUE), diabetes_data_copy$Glucose)

# Replace missing values with mean for 'BloodPressure' column
diabetes_data_copy$BloodPressure <- ifelse(is.na(diabetes_data_copy$BloodPressure), mean(diabetes_data_copy$BloodPressure, na.rm = TRUE), diabetes_data_copy$BloodPressure)

# Replace missing values with median for 'SkinThickness' column
diabetes_data_copy$SkinThickness <- ifelse(is.na(diabetes_data_copy$SkinThickness), median(diabetes_data_copy$SkinThickness, na.rm = TRUE), diabetes_data_copy$SkinThickness)

# Replace missing values with median for 'Insulin' column
diabetes_data_copy$Insulin <- ifelse(is.na(diabetes_data_copy$Insulin), median(diabetes_data_copy$Insulin, na.rm = TRUE), diabetes_data_copy$Insulin)

# Replace missing values with median for 'BMI' column
diabetes_data_copy$BMI <- ifelse(is.na(diabetes_data_copy$BMI), median(diabetes_data_copy$BMI, na.rm = TRUE), diabetes_data_copy$BMI)

# NOW, lets plot the new distribtuions 

# Setting up the plotting area of 1x1 gird 
par(mfrow=c(1, 1))  

# Iterating through each column and plotting histograms
for (col in names(diabetes_data_copy)) {
  hist(diabetes_data_copy[[col]], 
       main = paste("Histogram of", col),
       xlab = col,
       ylab = "Frequency",
       col = "skyblue",
       border = "black")
}

# we removed a tricep skinthickness of 99mm as it is an impossible value.   
diabetes_data_new <- diabetes_data_copy %>%
  filter(SkinThickness != 99)
summary(diabetes_data_new)

# Now that we have cleaned the data, we will create a scatterplot to analyse the data further
pairs(~ . , panel=panel.smooth, data = diabetes_data_new, main = "Scatterplot Matrix of Diabetes Data")

# Create a correlation matrix of the numeric columns of the dataframe
cor_matrix <- cor(diabetes_data_new[, sapply(diabetes_data_new, is.numeric)])

# Plot the correlation matrix using ggcorrplot
ggcorrplot(cor_matrix, type = "upper", lab = TRUE )

#-----------------------------------------------------------------------------------------------------------

#====================================== Unsupervised Learning Technique ====================================

# --- K Means Clustering ----

# Substantive issue : Is Bloodpressure a significant differentiator for Glucose ?

# Split the data into predictors (X) and the target variable (y)
# Exclude the last column (Outcome) as predictors
X <- diabetes_data_new[, -ncol(diabetes_data_new)]
y <- diabetes_data_new$Outcome

set.seed(2024)
scaled_X <- scale(X)

# set k = 2 to see a natural cluster of 2
k2 <- kmeans(scaled_X, centers=2) 
summary(k2)
k2results <- data.frame(diabetes_data_new$Glucose, diabetes_data_new$BloodPressure, k2$cluster)
cluster1 <- subset(k2results, k2$cluster==1)
cluster2 <- subset(k2results, k2$cluster==2)

summary(cluster1$diabetes_data_new.BloodPressure)
summary(cluster2$diabetes_data_new.BloodPressure)
## Cluster 2 has higher BloodPressure than cluster 1.

round(prop.table(table(cluster1$diabetes_data_new.Glucose)),5)
round(prop.table(table(cluster2$diabetes_data_new.Glucose)),5)
## cluster 2 has women with higher glucose levels while cluster 1 has women with lesser glucose levels. 

# Perform independent samples t-test
t_test_result <- t.test(cluster1$diabetes_data_new.Glucose, 
                        cluster2$diabetes_data_new.Glucose)

# Print the results
print(t_test_result)

# Using Welch 2 sample t-test, the mean glucose level of cluster 2 is higher than cluster 1. 
# Hence suggesting that women with higher glucose levels tend to have higher blood pressure as well. 

# Plot the clusters
plot(scaled_X, col = k2$cluster, 
     main = "K-means Clustering",
     xlab = "Glucose", ylab = "BloodPressure")

# Add centroids to the plot
points(k2$centers, col = 1:2, pch = 8, cex = 2)

# As a result, bloodpressure is a significant differentiator for Glucose. 

# ---PCA---  

PimaIndiansDiabetes <- diabetes_data_new[, -9]
str(PimaIndiansDiabetes)
apply(PimaIndiansDiabetes, 2, mean)
apply(PimaIndiansDiabetes, 2, var)
pr_scaled=prcomp(PimaIndiansDiabetes, scale=TRUE)
pr_scaled$rotation
dim(pr_scaled$x)
biplot(pr_scaled, scale=0)
pr_scaled$rotation=-pr_scaled$rotation
pr_scaled$x=-pr_scaled$x
biplot(pr_scaled, scale=0)
pr_scaled$sdev
pr.var=pr_scaled$sdev^2
pr.var
pve=pr.var/sum(pr.var)
pve
par(mfrow=c(1,2))
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b', col = "blue", lwd = 1.5)
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b', col = "blue", lwd= 1.5)
par(mfrow=c(1,1))

pc <- prcomp(PimaIndiansDiabetes, scale.=T) 
summary(pc)
#First two principal components capture 47.4% of variance. 


#======================================== Classification Technique ========================================

# --- KNN  ----

# Substantive issue : We want to find the optimal K value and its accuracy. 

# Split the data into training and testing sets. Using the 70-30 split ratio.
set.seed(2024)  
train_index <- sample(1:nrow(diabetes_data_new), 0.7 * nrow(diabetes_data_new))
X_train <- scaled_X[train_index, ]
y_train <- y[train_index]
X_test <- scaled_X[-train_index, ]
y_test <- y[-train_index]


Accuracy <- numeric(20)
for (i in 1:20) {
  knn_pred <- knn(train = X_train, test = X_test, cl = y_train, k = i)
  Accuracy[i] <- sum(knn_pred == y_test) / length(y_test)
}

plot(1:20, Accuracy, type = "b", xlab = "K", ylab = "Accuracy", main = "Accuracy vs K", col = 'orange', lwd = 1)

optimal_k <- which.max(Accuracy)
cat("Optimal value of K:", optimal_k, "\n")
# The optimal value achieved is k=16 

k <- 16  
knn_model <- knn(train = X_train, test = X_test, cl = y_train, k = k )

confusion_matrix <- table(y_test, knn_model)
cat("Confusion Matrix:\n", confusion_matrix, "\n")

confusionMatrix(confusion_matrix)

# Evaluate the model
Accuracy <- sum(knn_model == y_test) / length(y_test)
print(paste("Accuracy:", Accuracy))
# The accuracy when k=16 is 80%

# Calculate precision score
precision_score <- function(y_true, y_pred, positive_class = 1) {
  # Count true positives (TP) and false positives (FP)
  TP <- sum(y_true == positive_class & y_pred == positive_class)
  FP <- sum(y_true != positive_class & y_pred == positive_class)
  
  # Calculate precision
  precision <- TP / (TP + FP)
  return(precision)
}

# Calculate recall score
recall_score <- function(y_true, y_pred, positive_class = 1) {
  # Count true positives (TP) and false negatives (FN)
  TP <- sum(y_true == positive_class & y_pred == positive_class)
  FN <- sum(y_true == positive_class & y_pred != positive_class)
  
  # Calculate recall
  recall <- TP / (TP + FN)
  return(recall)
}

# Calculate F1 score
f1_score <- function(y_true, y_pred, positive_class = 1) {
  # Calculate precision and recall
  precision <- precision_score(y_true, y_pred, positive_class)
  recall <- recall_score(y_true, y_pred, positive_class)
  
  # Calculate F1 score
  f1 <- 2 * precision * recall / (precision + recall)
  return(f1)
}

# Calculate scores for the positive class (Outcome = 1)
precision <- precision_score(y_test, knn_model, positive_class = 1)
recall <- recall_score(y_test, knn_model, positive_class = 1)
f1 <- f1_score(y_test, knn_model, positive_class = 1)

cat("Precision Score:", precision, "\n")
cat("Recall Score:", recall, "\n")
cat("F1 Score:", f1, "\n")


# Perform KNN classification
k <- 16  
knn_model <- knn(train = X_train, test = X_test, cl = y_train, k = k, prob = TRUE)


# Calculate the predicted probabilities for the positive class (Outcome = 1)
probabilities <- attr(knn_model, "prob")

# Generate ROC curve
y_train <- factor(y_train)
y_test <- factor(y_test)
roc_curve <- roc(y_test, probabilities, levels = c(0, 1))
plot(roc_curve, col = 'blue', main = "ROC Curve for KNN Model", print.auc = TRUE)



# --- Logistic Regression ----

# Substantive issue : How does glucose level affect the probability of getting diabetes ?

set.seed(2024)

# Perform train-test split
train_index <- createDataPartition(diabetes_data_new$Outcome, p = 0.7, list = FALSE)
train_data <- diabetes_data_new[train_index, ]
test_data <- diabetes_data_new[-train_index, ]

# Fit logistic regression model using the training data
logistic_model <- glm(Outcome ~ Glucose, data = train_data, family = binomial)
summary(logistic_model)
# Z = -5.603767 + 0.039581(Glucose)

OR <- exp(coef(logistic_model))
OR

OR.CI <- exp(confint(logistic_model))
OR.CI

# Create a sequence of values for the predictor variable
glucose_values <- seq(min(diabetes_data_new$Glucose), max(diabetes_data_new$Glucose), length.out = 500)

# Predict probabilities using the fitted model
predicted_probabilities <- predict(logistic_model, newdata = data.frame(Glucose = glucose_values), type = "response")

# Combine predictor values and predicted probabilities into a data frame
plot_data <- data.frame(Glucose = glucose_values, Predicted_Probability = predicted_probabilities)

# Plot the logistic regression curve
ggplot(train_data, aes(x = Glucose, y = Outcome)) +
  geom_point(alpha = 0.5) +
  geom_line(data = plot_data, aes(x = Glucose, y = Predicted_Probability), color = "pink", size = 1) +
  labs(title = "Log Reg for Diabetes Prediction",
       x = "Glucose Level",
       y = "Probability of Diabetes") +
  theme_minimal()


# substantive issue :Using logistic regression, we then examined the statistically significant of the 8 input variables along with its accuracy. 
# Fit a full logistic regression model using the training data
Full_logistic_model <- glm(Outcome ~ ., data = train_data, family = binomial)
summary(Full_logistic_model)

# Predict probabilities using the fitted model on test data
predicted_probabilities <- predict(Full_logistic_model, newdata = test_data, type = "response")

# Convert probabilities to binary predictions (0 or 1) using a threshold of 0.5
binary_predictions <- ifelse(predicted_probabilities >= 0.5, 1, 0)

# Calculate confusion matrix
LR_conf_matrix <- table(test_data$Outcome, binary_predictions)
LR_conf_matrix

# Calculate accuracy
log_reg_accuracy <- sum(diag(LR_conf_matrix)) / sum(LR_conf_matrix)
log_reg_accuracy



# ---Naive Bayes classification technique--- 

# substantive issue : To evaluate the model’s performance using accuracy, precision, recall score and f1 score together with generating an ROC curve.

# Train the Naive Bayes classifier using the training data
model <- naiveBayes(Outcome ~ ., data = train_data)

# Make predictions on the test data
predictions <- predict(model, test_data[, -9])  # Exclude the Outcome column from the test data

# Evaluate the accuracy of the model
NB_accuracy <- mean(predictions == test_data$Outcome)
print(paste("Accuracy:", NB_accuracy))
# The accuracy is 76.52 %

train_data$Outcome <- factor(train_data$Outcome)
test_data$Outcome <- factor(test_data$Outcome)

conf_matrix <- confusionMatrix(predictions, test_data$Outcome)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

NB_precision <- conf_matrix$byClass["Precision"]
NB_recall <- conf_matrix$byClass["Recall"]
NB_f1_score <- conf_matrix$byClass["F1"]

# Print precision, recall, and F1-score
print(paste("NB_Precision:", NB_precision))
print(paste("NB_Recall:", NB_recall))
print(paste("NB_f1-Score:", NB_f1_score))


# Plot ROC curve
roc_curve_NB <- roc(test_data$Outcome, as.numeric(predictions))
plot(roc_curve_NB, main = "ROC Curve for Naive Bayes Classifier", col = "blue")

NB_auc_value <- auc(roc_curve_NB)
print(paste("AUC:", NB_auc_value))

# ---Decision tree classification method--- 

# substantive issue:To evaluate the accuracy of the decision tree model to identify patients with and without diabetes. 

# Build the decision tree model using the training data
DT_model <- rpart(Outcome ~ ., data = train_data, method = "class")
DT_predictions <- predict(DT_model, test_data, type = "class")

# Calculate accuracy
DT_accuracy <- mean(DT_predictions == test_data$Outcome)

# Print accuracy
print(paste("Accuracy:", DT_accuracy))

# To make the decision tree look nicer we will plot using rpart. 
model <- rpart(Outcome ~ ., data = train_data, method = "class")
rpart.plot(model, type = 2, extra = 101, cex = 0.5)


# ---svm model---
# substantive issue: To evaluate the model’s performance using accuracy, precision, recall score and f1 score together with generating an ROC curve.

# Train the SVM model
svm_model <- svm(Outcome ~ ., data = train_data, kernel = "radial", type = 'C-classification')

# Make predictions on the test set
predictions_svm <- predict(svm_model, test_data[,-ncol(test_data)])  

# Evaluate the model
svm_accuracy <- mean(predictions_svm == test_data$Outcome)
print(paste("Accuracy:", svm_accuracy))
# accuracy = 0.79130 (5sf)

# Calculate F1 score, recall score as well as precision score 
svm_f1 <- f1_score(test_data$Outcome, predictions_svm)
svm_recall <- recall_score(test_data$Outcome, predictions_svm)
svm_precision <- precision_score(test_data$Outcome, predictions_svm)

cat("SVM Model Evaluation:\n")
cat("F1 Score:", svm_f1, "\n")
cat("Recall Score:", svm_recall, "\n")
cat("Precision Score:", svm_precision, "\n")

# Calculate AUC
auc_SVM <- roc(test_data$Outcome, as.numeric(predictions_svm))
print(auc_SVM)

# Plot ROC curve
plot(auc_SVM, main = " SVM ROC Curve", col = "blue")

auc_value <- auc(auc_SVM)
print(paste("AUC:", auc_value))

#========================================== Regression Technique ==========================================
# ---Linear Regression--- 

# substantive issue : we want to test the effect glucose has on bloodpressure.

linear_model <- lm(BloodPressure ~ Glucose, data = diabetes_data_new)
summary(linear_model)
model <- ("Linear Reg")
# Plotting the regression line
plot(diabetes_data_new$Glucose, diabetes_data_new$BloodPressure,
     xlab = "Glucose", ylab = "Blood Pressure", main = "Effect of Glucose on Blood Pressure",
     col = "skyblue", pch = 19, lwd = 0.5 )
abline(linear_model, col = "black", lwd = 2 )

# RMSE of linear regression 

full_linear_model <- lm(Glucose ~ ., data = train_data)
lr_predictions <- predict(full_linear_model, newdata = test_data)
lr_residuals <- test_data$Glucose - lr_predictions

# Calculate the RMSE
LR_RMSE <- sqrt(mean(lr_residuals^2))
print(paste("RMSE for Linear Regression (Test Data):", LR_RMSE))



# ---Random Forest--- 

# Substantive issue: To calculate the importance of 7 other input variables where the random forest outcome is glucose. 
Rand_forest_model <- randomForest(Glucose ~ . , data=train_data[,-9])
Rand_forest_model

importance <- importance(Rand_forest_model)
print(importance)
varImpPlot(Rand_forest_model, col = "red")

# OOB RMSE
OOB_RMSE <- sqrt(Rand_forest_model$mse[Rand_forest_model$ntree])
OOB_RMSE
plot(Rand_forest_model)
## This Confirms that the error stablised before 500 trees.

print(paste("RMSE for Random Forest:", OOB_RMSE))
print(paste("RMSE for Linear Regression:",LR_RMSE ))


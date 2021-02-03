setwd('../iamchetry/Documents/UB_files/506/hw_7/')

#install.packages('neuralnet')
#install.packages('fastDummies')
#install.packages('plotly')

library(rpart) 
library(MASS)
library(caret)
library(dplyr)
library(glue)
library(leaps)
library(pROC)
library(randomForest)
library(neuralnet)
library(fastDummies)
library(plotly)


#----------------------- 1st Question ------------------------

load('cleveland.RData')
data_ = cleveland
data_ = data_[, -15]
attach(data_)

set.seed(10)
t = createDataPartition(diag1, p=0.7, list = FALSE)
train_ = na.omit(data_[t, ])
test_ = na.omit(data_[-t, ])
y_true = test_$diag1

#CART
control_ = rpart.control(minsplit = 10, xval = 5, cp = 0)
tree_ = rpart(diag1~., data = train_, method = "class", control = control_)

plot(tree_$cptable[,4], main = "Cp for model selection", ylab = "Cp", type='line')

min_cp = which.min(tree_$cptable[,4])
pruned_tree = prune(tree_, cp = tree_$cptable[min_cp,1])

#Feature Importance
plot(pruned_tree$variable.importance, xlab="variable", 
     ylab="Importance", xaxt = "n", pch=20)
axis(1, at=1:length(pruned_tree$variable.importance), 
     labels=names(pruned_tree$variable.importance))

par(mfrow = c(1,2))
plot(pruned_tree, branch = .3, compress=T, main = "Pruned Tree")
text(pruned_tree, cex = .5)

plot(tree_, branch = .3, compress=T, main = "Full Tree")
text(tree_, cex = .5)

my_pred = predict(pruned_tree, newdata = test_, type = "class")

tab_test = table(my_pred, y_true)
conf_test = confusionMatrix(tab_test)

test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error


# Random Forest
rf_model = randomForest(diag1~., data = train_, ntree = 500, mtry = 8)
#par(mfrow = c(1,2))
varImpPlot(rf_model, main='Feature Importances')
importance(rf_model)

rf_pred = predict(rf_model, newdata = test_, type = "response")
tab_test = table(rf_pred, y_true)
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error


# Neural Net
train_dummied = fastDummies::dummy_cols(train_)
test_dummied = fastDummies::dummy_cols(test_)
train_dummied = train_dummied[, -c(2, 3, 6, 7, 9, 11, 13, 34, 35)]
test_dummied = test_dummied[, -c(2, 3, 6, 7, 9, 11, 13, 34, 35)]

#1 Hidden Node
nn0 = neuralnet(diag1~., data = train_dummied, hidden = 1, err.fct = "ce",
                stepmax = 10^9, linear.output = FALSE)
plot(nn0)
preds = predict(nn0, newdata=test_dummied)
preds = round(preds)[, 2]

pred_labels = c()
for (i in preds)
{
  if (i == 1)
  {
    pred_labels = c(pred_labels, 'sick')
  }
  else
  {
    pred_labels = c(pred_labels, 'buff')
  }
}

tab_test = table(pred_labels, y_true)
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error

#2 Hidden Nodes
nn1 = neuralnet(diag1~., data = train_dummied, hidden = 2, err.fct = "ce",
                stepmax = 10^9, linear.output = FALSE)
plot(nn1)
preds = predict(nn1, newdata=test_dummied)
preds = round(preds)[, 2]

pred_labels = c()
for (i in preds)
{
  if (i == 1)
  {
    pred_labels = c(pred_labels, 'sick')
  }
  else
  {
    pred_labels = c(pred_labels, 'buff')
  }
}

tab_test = table(pred_labels, y_true)
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error

#3 Hidden Nodes
nn2 = neuralnet(diag1~., data = train_dummied, hidden = 3, err.fct = "ce",
                stepmax = 10^9, linear.output = FALSE)
plot(nn2)
preds = predict(nn2, newdata=test_dummied)
preds = round(preds)[, 2]

pred_labels = c()
for (i in preds)
{
  if (i == 1)
  {
    pred_labels = c(pred_labels, 'sick')
  }
  else
  {
    pred_labels = c(pred_labels, 'buff')
  }
}

tab_test = table(pred_labels, y_true)
conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error


#------------------------ 2nd Question ------------------------------

load('pendigits.RData')
data_ = pendigits
attach(data_)

set.seed(20)
t = createDataPartition(V36, p=0.7, list = FALSE)
train_ = na.omit(data_[t, ])
test_ = na.omit(data_[-t, ])
y_true = test_$V36

YY = as.factor(V36)
features_ = c(1:35)
variances_ = c()

for (i in 1:35)
{
  variances_ = c(variances_, var(data_[, i]))
}

plot(features_, variances_, main='Feature-wise Variances', xlab='Feature',
     ylab='Variance', type='p', col='red', pch='*')

pc_ex = prcomp(data_[, 1:35], center = FALSE, scale = TRUE)
plot(pc_ex)

pc_var = (pc_ex$sdev)^2
per_var_exp = (pc_var/(sum(pc_var)))*100
barplot(per_var_exp, main = "PC variance explained", ylab = "% variation explained", xlab = "PCs")

par(mfrow = c(1,3))
plot(pc_ex$x[,1], pc_ex$x[,2], xlab = "PC1 scores", ylab = "PC2 scores")
plot(pc_ex$x[,1], pc_ex$x[,3], xlab = "PC1 scores", ylab = "PC3 scores")
plot(pc_ex$x[,2], pc_ex$x[,3], xlab = "PC2 scores", ylab = "PC3 scores")

plot_ly(x=pc_ex$x[,1], y=pc_ex$x[,2], z=pc_ex$x[,3], type="scatter3d", mode="markers", color=YY)

#KNN
require(class)


# KNN on Raw Data --------------------------
K = c()
E_tr = c()
E_ts = c()

for (k in seq(1, 101, 2))
{
  KNN_train = knn(train_[-c(36)], train_[-c(36)], train_$V36, k) # Train Prediction
  KNN_test = knn(train_[-c(36)], test_[-c(36)], train_$V36, k) # Test Prediction
  
  train_predicted = as.factor(KNN_train)
  test_predicted = as.factor(KNN_test)
  
  tab_train = table(train_predicted, as.factor(train_$V36))
  tab_test = table(test_predicted, as.factor(test_$V36))
  
  #-------- Confusion Matrix for Accuracy ---------
  conf_train = confusionMatrix(tab_train)
  conf_test = confusionMatrix(tab_test)
  
  K = c(K, k)
  E_tr = c(E_tr, 1 - round(conf_train$overall['Accuracy'], 4))
  E_ts = c(E_ts, 1 - round(conf_test$overall['Accuracy'], 4))
}

par(mfrow = c(2, 1))
plot(K, E_tr, main = 'Train Performance : Choosing different values of k',
     xlab = 'K', ylab = 'Train Error', col='blue', type='line')
plot(K, E_ts, main = 'Test Performance : Choosing different values of k',
     xlab = 'K', ylab = 'Test Error', col='blue', type='line')


# KNN on Principal Components ----------------------
pc_train = prcomp(train_[, 1:35], center = FALSE, scale = TRUE)
plot(pc_train)
pc_var = (pc_train$sdev)^2
per_var_train = (pc_var/(sum(pc_var)))*100

pc_test = prcomp(test_[, 1:35], center = FALSE, scale = TRUE)
plot(pc_test)
pc_var = (pc_test$sdev)^2
per_var_test = (pc_var/(sum(pc_var)))*100

K = c()
E_tr = c()
E_ts = c()

for (k in seq(1, 101, 2))
{
  KNN_train = knn(pc_train$x[, 1:2], pc_train$x[, 1:2], train_$V36, k) # Train Prediction
  KNN_test = knn(pc_train$x[, 1:2], pc_test$x[, 1:2], train_$V36, k) # Test Prediction
  
  train_predicted = as.factor(KNN_train)
  test_predicted = as.factor(KNN_test)
  
  tab_train = table(train_predicted, as.factor(train_$V36))
  tab_test = table(test_predicted, as.factor(test_$V36))
  
  #-------- Confusion Matrix for Accuracy ---------
  conf_train = confusionMatrix(tab_train)
  conf_test = confusionMatrix(tab_test)
  
  K = c(K, k)
  E_tr = c(E_tr, 1 - round(conf_train$overall['Accuracy'], 4))
  E_ts = c(E_ts, 1 - round(conf_test$overall['Accuracy'], 4))
}

par(mfrow = c(2, 1))
plot(K, E_tr, main = 'Train Performance for PCs : Choosing different values of k',
     xlab = 'K', ylab = 'Train Error', col='blue', type='line')
plot(K, E_ts, main = 'Test Performance for PCs : Choosing different values of k',
     xlab = 'K', ylab = 'Test Error', col='blue', type='line')


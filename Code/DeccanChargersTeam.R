library (VIM) # for calculating missing values
library (MASS) #for stepAIC
library(pls)  #load the plsr package for Partial Least Squares Regression
library(car)  #load car to get vif
library(caret) # for predict function
library (magrittr)
library(lars)
library(glmnet)# For LASSO
library(randomForest)
library(mice)
library(magrittr)
library(pls)
library(ISLR)
library(e1071)
library(modelr)
library(AppliedPredictiveModeling)
library(lars)
library(pls)
library(elasticnet)
library(gbm)

#Read the train/test file
hdata <- read.csv("../../Assignment 4/Homework4/housingData2.csv")
hdata_t <- read.csv("../../Assignment 4/Homework4/housingTest2.csv")


#Remove the variables "BsmtUnfSF", "LowQualFinSF"
hdata <- hdata[,setdiff(names(hdata), c( "BsmtUnfSF", "LowQualFinSF"))]
hdata_t <- hdata_t[,setdiff(names(hdata_t), c( "BsmtUnfSF", "LowQualFinSF"))]


#Storing The ID field in seperate vector
Id <- hdata_t$Id

hdata <- hdata[,-c(1,2,74)] # removing X,Id, X.1 from housing traiing data
hdata_t <- hdata_t[,-c(1,2)] # removing X,Id, X.1 from housing test data

#Storing The price field in seperate vector
saleprice <- hdata[,71]
hdata <- hdata[,-71] # removing sales price from training data
features <- rbind(hdata,hdata_t) #combining training and test data
features1 <- features


# Function to Handle Missing value  using median and Change of Not applicable variables to "none"
handle_na <- function(features_df){
  for(i in 1:ncol(features_df)){
    item <- features_df[,i]
    if(is.numeric(item)){
      features_df[is.na(item),i] <- median(features_df[!is.na(item),i]) #Add median to the numerical variables
    } else{
      item <- levels(item)[item]
      item[is.na(item)] <- "none" #a NA becomes a new category
      features_df[,i] <- as.factor(item)
    }
  }
  features_df
}

#Handle NA's
features <- handle_na(features)
#we can also handle NA's using mice package but as per our understanding handling NA's manually is the better option.

levels(features$KitchenQual) <- c(4,1,3,2,2)
features$KitchenQual <- as.numeric(levels(features$KitchenQual)[features$KitchenQual])
levels(features$ExterQual) <- c(4, 1, 3, 2)
features$ExterQual <- as.numeric(levels(features$ExterQual)[features$ExterQual])

anyNA(features) #checking NA's
#Applying Log transformation to the sale price
LogPrice <- log10(saleprice)
anyNA(LogPrice) #checking NA's in sale price

numerical_var = names(hdata)[which(sapply(hdata, is.numeric))]
housing_cor_numerics = cor(na.omit(hdata[,numerical_var]))
col <- colorRampPalette(c("#BB4444","#EE9988","#FFFFFF","#77AADD","#4477AA"))
corrplot:: corrplot(housing_cor_numerics, method="circle", insig = "blank",shade.col=NA,tl.srt=45)

#adding levels to make all test and train data should have same levels
levels(features$MSZoning) <- c(levels(features$MSZoning),"None")
levels(features$Exterior1st) <- c(levels(features$Exterior1st),"None","ImStucc","Stone")
levels(features$Exterior2nd) <- c(levels(features$Exterior2nd),"None","Other")
levels(features$KitchenQual) <- c(levels(features$KitchenQual),"None")
levels(features$Functional) <- c(levels(features$Functional),"None")
levels(features$SaleType) <- c(levels(features$SaleType),"None")

#Skew Transformation for Numerical Variables
# Taking all the column classes in one variable so as to seperate factors from numerical variables
Column_classes <- sapply(names(features),function(x){class(features[[x]])})
numeric_columns <-names(Column_classes[Column_classes != "factor"])

#determining skew of each numric variable
skew <- sapply(numeric_columns,function(x){skewness(features[[x]],na.rm = T)})

# Let us determine a threshold skewness and transform all variables above the treshold.
skew <- skew[skew > 0.75]

# transform excessively skewed features with log(x + 1)
for(x in names(skew)) {
  features[[x]] <- log(features[[x]] + 1)
}
LogPrice <- log10(saleprice)
print("Yuppiee Everything looks good till now")

# OLS
housedata_val <- features[1:100,]
housedata_train <- features[101:1000,]
saleprice_val <- saleprice[1:100]
saleprice_train <- saleprice[101:1000]
#
fit <- lm(log(saleprice_train)~ ., housedata_train)
summary(fit)
AIC(fit) # -624.6101
BIC(fit) # 186.9947
# R Squared value is good. All are not passed Hypothessis test. Few values have >0.05. so this model is
#not good. potential overfitting could occur if someone insist on using it. The vaiable selection process
#should be invloved in model construction. I prefer to use stepAIC method.
fit2 <- stepAIC(fit, direction = "both")
summary(fit2)
AIC(fit2) # -690.3082
BIC(fit2) # -116.9265
vif(fit2)
#The R Square is good, and all variables pass the Hypothesis Test. The diagonsis of residuals is also better. 
#this gives r- square value as 0.9091 for best fit

plot(fit2)
RSS <- c(crossprod(fit2$residuals))
MSE <- RSS / length(fit2$residuals) #Mean squared error:
RMSE <- sqrt(MSE) #0.1451161        #Root Mean Squared Error

pred <- predict(fit2,newdata = housedata_val)
plot(pred, housedata_val$SalePrice)

res <- resid(fit2)
#We now plot the residual against the observed values of the variable waiting.
plot(saleprice_train,res,ylab="Residuals", xlab="Sale Price", main="Residuals vs SalePrice") 

# Almost all data variables has outliers (MSSubClass, LotArea, overall, YearsUsed, MassVnrArea, TotalBsmtSF,Floorsqft,GarageCars)
influencePlot(fit2)

#--------------------------------------------------------
#PLS

#loading Train data and Test data to new variables
housedata_pls <- features[1:1000,]
housedata_test <- features[1001:1179,]

#Log Transformation for the Price Variable and Fitting the Model using Cross validation
plsfit <- plsr(LogPrice~.,data=housedata_pls, validation = "CV")

#Predicting the Fitted model using first two components
plspred <- predict(plsfit,housedata_test,ncomp=1:2)

#Transforming the log transformed Price vaue to its original value
plspredsaleprice <- round(10.0**plspred)


#Plotting the PLS fit and validation plots to find the RMSEP value
plot(plsfit, main = "RMSE PLS Price Prediction", xlab="components")
validationplot(plsfit,val.type = "RMSEP")

#Hypertuning: Hyper parameter tuning using the components obtained corresponding to RMSEP
pls.RMSEP <- RMSEP(plsfit,estimate = "CV")
plot(pls.RMSEP, main = "RMSE PLS Price Prediction", xlab="components")
min <- which.min(pls.RMSEP$val)
points(min,min(pls.RMSEP$val),pch=1,col="blue",cex=1.5)
min


#Predicting using the 30 components
plspred_tuned <- predict(plsfit,housedata_test,ncomp=30)
plspredsaleprice_tuned <- round(10.0**plspred_tuned)
plot(plsfit,ncomp=30,asp=1,line=TRUE)

submission_plsskew <- data.frame(Id, plspredsaleprice_tuned)
write.csv(submission_plsskew, "submission_plsskew.csv", row.names = F)

#-------------------------------------------------------------

#LASSO
#loading Train data and Test data to new variables
housedata_lasso <- features[1:1000,]
housedata_lasso_test <- features[1001:1179,]

#Converting the factor variables in the train and test data to numeric
for(i in 1:ncol(housedata_lasso)){
  if(is.factor(housedata_lasso[,i])){
    housedata_lasso[,i]<-as.numeric(housedata_lasso[,i])
  }
}
for(i in 1:ncol(housedata_lasso_test)){
  if(is.factor(housedata_lasso_test[,i])){
    housedata_lasso_test[,i]<-as.numeric(housedata_lasso_test[,i])
  }
}

#write.csv(submission_lassoskew, "submission_lassoskew.csv", row.names = F)
set.seed(1)

#Converting the log of dependent variable to vector and independent variable to matrix
y <- as.numeric(LogPrice)      #target variable
x <- as.matrix(housedata_lasso)     #predictors

cvLasso <- cv.glmnet(x,y,alpha=1) # LASSO USING glmnet
plot.cv.glmnet(cvLasso)
bestlam <- cvLasso$lambda.min
bestlam

#Repeated Cross Validation & HyperParameter Tuning with caret package with mygrid
#Custom Grid created for LASSO Regression
myGrid <- expand.grid(alpha=1,  # Lasso regression
                     lambda=c(1,0.1,0.05,0.01,seq(0.009,0.001,-0.001),
                              0.00075,0.0005,0.0001))
set.seed(123) 
CARET.TRAIN.CTRL <- trainControl(method="repeatedcv",
                                 number=5,
                                 repeats=5,
                                 verboseIter=FALSE)
model_lasso <- train(x = housedata_lasso,y = LogPrice,
                     method="glmnet",
                     metric="RMSE",
                     maximize=FALSE,
                     trControl=CARET.TRAIN.CTRL,
                     tuneGrid=myGrid)

#predicting prices using repeated cv lasso model
lassopred <- predict(model_lasso,newdata=housedata_lasso_test)
lassopredsaleprice  <- round(10.0**lassopred)
submission_lassoskew <- data.frame(Id, lassopredsaleprice)
write.csv(submission_lassoskew, "submission_lassoskew1.csv", row.names = F)
plot(model_lasso)

#lasso performs even better so we'll just use this one to predict on the test set. 
#Another neat thing about the Lasso is that it does feature selection for you - 
#setting coefficients of features it deems unimportant to zero. Let's take a look at the coefficients:
#coefficients
coef <- data.frame(coef.name = dimnames(coef(model_lasso$finalModel,s=model_lasso$bestTune$lambda))[[1]], 
                   coef.value = matrix(coef(model_lasso$finalModel,s=model_lasso$bestTune$lambda)))

picked_features <- sum(coef$coef.value!=0)  #This will take good features coefficient not equal to zero
not_picked_features <- sum(coef$coef.value==0) # non picked features or bad features stores there

cat("Lasso picked",picked_features,"variables and eliminated the other",
    not_picked_features,"variables\n")

coef <- arrange(coef,-coef.value)

# extract the top 12 and bottom 12 features
imp_coef <- rbind(head(coef,12),
                  tail(coef,12))

#ggplot  for the 24 importantcoefficients
ggplot(imp_coef) +
  geom_bar(aes(x=reorder(coef.name,coef.value),y=coef.value),
           stat="identity") +
  ylim(-1.5,0.6) +
  coord_flip() +
  ggtitle("Coefficents in the Lasso Model") +
  theme(axis.title=element_blank())
#The most important positive feature is GrLivArea - the above ground area by area square feet. This definitely sense. Then a few other location and quality features contributed positively. Some of the negative features make less sense and would be worth looking into more - it seems like they might come from unbalanced categorical variables.
tune_value <- model_lasso$finalModel$tuneValue
print(tune_value)
#-------------------------------------------------------------------
#SVM(Support Vector Machines)

#loading Train data and Test data to new variables
housedata_svm <- features[1:1000,]
housedata_svm_test <- features[1001:1179,]

#Fitting the SVM Model for the Log of Price of houses using cost of constraints violation = 3.
svmfit <- svm(LogPrice ~ ., data = housedata_svm, cost = 3)

#Predicting the Outcome for the test data.
svmpred <- predict(svmfit, newdata = housedata_svm_test)

#Transforming the log transformed Price vaue to its original value
svmpredsaleprice <- round(10.0**svmpred)

tuneResult <- tune(svm, y ~ x,  data = housedata_svm,
                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9))
)
print(tuneResult)
# Draw the tuning graph
tunedModel <- tuneResult$best.model
tunedModelY <- predict(tunedModel, housedata_svm) 
plot(tuneResult)
points(housedata_svm, tunedModelY, col = "red", pch=4)

submission_svmskew <- data.frame(Id, svmpredsaleprice)
#write.csv(submission_svmskew, "submission_svmskew.csv", row.names = F)

#-------------------------------------

#Gradient boosting method(GBM)

#loading Train data and Test data to new variables
housedata_gbm <- features[1:1000,]
housedata_gbm_test <- features[1001:1179,]

#Fitting the Gbm Model for the Log of Price of houses using 300 trees.
gbmfit <- gbm(LogPrice ~., data = housedata_gbm, distribution = "laplace",
             shrinkage = 0.05,
             interaction.depth = 5,
             bag.fraction = 0.66,
             n.minobsinnode = 1,
             cv.folds = 100,
             keep.data = F,
             verbose = F,
             n.trees = 300)

#Predicting the Sale price for the test data.
gbmpred <- predict(gbmfit, newdata = housedata_gbm_test, n.trees = 300)

#Transforming the log transformed Price vaue to its original value
gbmpredsaleprice <- round(10.0**gbmpred)
plot(gbmfit,i="LotArea") 
#Inverse relation with lstat variable
plot(gbmfit,i="X1stFlrSF") 

submission_gbmskew <- data.frame(Id, gbmpredsaleprice)
#write.csv(submission_gbm, "submission_gbm.csv", row.names = F)

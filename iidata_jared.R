# Expert Group:
# Question 2 (5 points): Build a model that classifies food by whether it is a sugary snack, dairies, fresh food or meats. # You do not need to use the entire dataset if your model is too computationally expensive, but you do need to demonstrate # that the model is reproducible. Report your error rate.

# Used an optimized xgboost model because tsne shows that the data is non-linearly separable.





library(dplyr)
require(xgboost)
require(Matrix)
require(data.table)
require(vcd)
library(car)
library(corrplot)
library(Rtsne)
library(caret)

# =======================================================================================
# Select data to use
# =======================================================================================
setwd("~/Downloads/")
data <- read.csv("ffclean6.csv")
names_to_keep <- names(data)[17:length(data)]

df <- data[,c(names_to_keep, "pnns_groups_1")]

df <- df %>% filter(!is.na(as.vector(pnns_groups_1)))
# select columns given in problem statement
main_categories <- c("Sugary snacks", "Milk and dairy products", "Fish Meat Eggs", "Fruits and vegetables")
df <- df %>% filter(pnns_groups_1 %in% main_categories)

# drop na values
df <- df[complete.cases(df),]





# =======================================================================================
# Make everything numeric
# =======================================================================================
label <- as.vector(df$pnns_groups_1)


# make features numeric
#X <- df[,-1]
#X <- sapply(X, as.numeric)

# one hot encode label
y <- as.numeric(as.factor(label)) 



X = df


#sparse_matrix <- sparse.model.matrix(pnns_groups_1~.-1, data = X)
#head(sparse_matrix)

label <- as.numeric(as.factor(label))

ndf <- cbind(X, label)
ndf <- ndf[,-12]

write.csv(ndf, file="../Desktop/iidata.csv")






# TSNE


tsne = Rtsne(as.matrix(ndf), check_duplicates=FALSE, pca=TRUE, 
             perplexity=30, theta=0.3, dims=2)

embedding = as.data.frame(tsne$Y)
embedding$Class = ndf$label
g = ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
  theme(plot.title = element_text(lineheight=2, face="bold"))+
  geom_point(size=2, alpha=1, shape=19) +
  guides(colour=guide_legend(override.aes=list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE 2D Embedding of School Value") +
  theme_light(base_size=20) +
  theme(axis.text.x=element_blank(),
        axis.text.y=element_blank()) + 
  theme_solarized() +
  scale_colour_manual(values = c("red","blue", "green","orange"))
print(g)
















# =====================================================================================
# Define test and training sets
# =====================================================================================
set.seed(1)
# Define datasets
ndf <- as.data.frame(ndf)
train_ind <- sample(nrow(ndf), floor(0.7*nrow(ndf)))
train <- ndf[train_ind,]
test <- ndf[-train_ind,]
# Get response labels
train_labels <- train$label
train <- train[-grep('label', colnames(train))]
test_labels <- test$label
test <- test[-grep('label', colnames(test))]
# convert data to matrix
train.matrix = as.matrix(train)
mode(train.matrix) == "numeric"
test.matrix = as.matrix(test)
mode(test.matrix) == "numeric"
# convert outcome from factor to numeric matrix 
#   bc xgboost takes multi-labels in [0, numOfClass)
y = as.matrix(as.integer(train_labels)) - 1

# =======================================================================================
#
# =======================================================================================
num.class = 4
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = 4,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

# =======================================================================================
#
# =======================================================================================
set.seed(1234)
# k-fold cross validation, with timing
nround.cv = 50
system.time( bst.cv <- xgb.cv(param=param, data=train.matrix, label=y, 
                              nfold=4, nrounds=nround.cv, prediction=TRUE, verbose=T) )




# Find best parameters for xgboost
# =====================================================================================
# XGBOOST 2: Optimizing XGBOOST
# =====================================================================================
# find parameters associated with minimum error, use to get best model
min.merror.idx = which.min(bst.cv$dt[, test.merror.mean]) # identify index of best classification
bst.cv$dt[min.merror.idx,]  # minimum error metrics
# predict 
pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
pred.cv = max.col(pred.cv, "last")

# confusion matrix
confusionMatrix(factor(y+1), factor(pred.cv))




system.time( bst <- xgboost(param=param, data=train.matrix, label=y, 
                            nrounds=min.merror.idx, verbose=1) )
# Use best model to predict 
pred <- predict(bst, test.matrix)  
head(pred, 10)  
# Get accurracy of best model
pred = matrix(pred, nrow=num.class, ncol=length(pred)/num.class)
pred = t(pred)
pred = max.col(pred, "last")
pred.char = toupper(letters[pred])
cat("Accuracy: ", mean(as.numeric((pred )== test_labels)))


model = xgb.dump(bst, with.stats=TRUE)
# get the feature real names
names = dimnames(train.matrix)[[2]]
# compute feature importance matrix
importance_matrix = xgb.importance(names, model=bst)
# plot feature importance
gp = xgb.plot.importance(importance_matrix)
print(gp) 











#Question 1 (5 points), for all groups:
#Examine the “Nutrition Score FR” variable. How many French and non-French products have a French nutrition score? 
#Is the French nutrition score biased against non-French food products? Explain.

# Yes it is. Because the data doesn't meet basic assumptions for traditional testing techniques (e.g. independence, 
# normally distributed residuals, homoscedasiticity, etc.), a permutation test was employed.
# This test was implemented from scratch ans is shown below.
# The p-value, .001, reveals that the data is different. The null here is that there is no difference among french nutrition.


french <- read.csv("ffclean6_isfrench.csv")

par(mfrow=c(1,1))



french$nutrition_score_fr_100g <- as.factor(as.numeric(french$nutrition_score_fr_100g))
french$is_french <- as.factor(as.numeric(french$is_french))

newdf <- data.frame(response = french$nutrition_score_fr_100g, trt = as.vector(french$is_french))

newdf <- newdf %>% filter(!is.na(trt), !is.na(response))

newdf$trt <- as.character(newdf$trt)

newdf <- newdf %>% filter(trt !="")

newdf$trt[newdf$trt=="False"] = "FALSE"
newdf$trt[newdf$trt=="True"] = "TRUE"

newdf$trt <- as.logical(newdf$trt)

boxplot(response ~ trt, data=newdf)


tTest <- function(data, hasTrtment){
  m         <- sum(hasTrtment) # Treatment group size
  n         <- sum(!hasTrtment) # Control group size
  
  # difference of means (assuming unequal variances):
  mean_diff <- diffMeans(data=data, hasTrt=hasTrtment)
  # denominator:
  trt_var   <- var(data[hasTrtment])/m
  ctrl_var  <- var(data[!hasTrtment])/n
  sd_diff   <- sqrt(trt_var + ctrl_var)
  # t-statistic: 
  t_stat    <- mean_diff / sd_diff
  # p-value:
  df        <- min(n-1,m-1)
  pval      <- 1-pt(t_stat, df)
  
  return( pval )
}

tTest(newdf$response, newdf$trt)








diffMeans <- function(data, hasTrt){
  # computes our test statistics: the difference of means
  # hasTrt: boolean vector, TRUE if has treatment
  test_stat <- mean(data[hasTrt]) - mean(data[!hasTrt])
  return(test_stat)
}

currentStat <- diffMeans(newdf$response, newdf$trt)

# ====================================================================================
# Step 7: Bootstrap
# ====================================================================================
simPermDsn <- function(data, hasTrt, testStat, k=100){
  # Simulates the permutation distribution for our data
  # hasTrt: boolean indicating whether group is treatment or control
  # testStat: function defining the test statistic to be computed
  # k: # of samples to build dsn.      
  sim_data   <- replicate(k, sample(data))
  test_stats <- apply(sim_data, 2,
                      function(x) {testStat(x, hasTrt)})
  return( test_stats)
}


permutationTest <- function(data, hasTrt, testStat, k=1000){
  # Performs permutation test for our data, given some pre-selected
  # test statistic, testStat
  currentStat    <- testStat(data, hasTrt)
  simulatedStats <- simPermDsn(data, hasTrt,testStat, k)
  
  # 2-way P-value
  pValue         <- sum(abs(simulatedStats) >= currentStat)  / k
  
  return(pValue)
}

permutation_results <- simPermDsn(data=newdf$response, 
                                  hasTrt=newdf$trt, 
                                  testStat=diffMeans,
                                  k=5000)

set.seed(619)
p.val <- permutationTest(newdf$response, newdf$trt, testStat = diffMeans)
cat("p-value: ", p.val)


hist(permutation_results)
abline(v=currentStat)

# same plot as above but much prettier
pr <- as.data.frame(permutation_results)
ggplot(pr, aes(x=permutation_results)) + 
  geom_histogram(fill='skyblue3', alpha=.8) +
  geom_vline(xintercept = currentStat, colour='tomato', alpha=.9, size=1) +
  theme_economist() + scale_color_economist() +
  labs(title="Distribution of Bootstrapped Test Statistics: French Difference in Nutrition", xlab = "Difference of Means", ylab="Count")













  # Question 5 (Free Response) (5 points): Create a statistical model that tries to explain a continuous variable. This can be linear regression, 
  # ANOVA or anything you want. You will have to justify and explain your model.

# ANSWER: For this question, we use carboyhdrates as a continuous response value. 
# I wrote a multivariable-linear regression model. Model diagnostics are shown below
# In explaining the continuous variable,
# we can see that it does not meet basic assumptions of linear models, so a linear model shouldnt
# be used in this case (without relevant transformations).


# code
# =======================================================================================
# Select data to use
# =======================================================================================

names_to_keep <- names(data)[17:length(data)]

df <- data[,c(names_to_keep, "pnns_groups_1")]

df <- df %>% filter(!is.na(main_category_en))
# select columns given in problem statement
main_categories <- c("Sugary snacks", "Milk and dairy products", "Fish Meat Eggs", "Fruits and vegetables")
df <- df %>% filter(pnns_groups_1 %in% main_categories)

# drop na values
df <- df[complete.cases(df),]

# =======================================================================================
# Make everything numeric
# =======================================================================================
label <- as.vector(df$pnns_groups_1)
# make features numeric
#X <- df[,-1]
#X <- sapply(X, as.numeric)
# one hot encode label
y <- as.numeric(as.factor(label)) 
X = df

#sparse_matrix <- sparse.model.matrix(pnns_groups_1~.-1, data = X)
#head(sparse_matrix)
label <- as.numeric(as.factor(label))
ndf <- cbind(X, label)
ndf <- ndf[,-12]
ndf_lm <- lm(carbohydrates_100g ~ ., data=ndf)
par(mfrow=c(2,2))
plot(ndf_lm)

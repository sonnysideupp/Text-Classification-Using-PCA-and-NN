load(paste("/Users/sonnyhuang/Downloads/","reuters.RData",sep=""))

#the dataframe contains 2000 news stories, and each news story 
#contains 11707 words 

#stories on coporate events are labeled as 1 and stories on economy
#are labeled as 0 

#40% training data and 60% testing data
first800stories = head(rt.bow,800)
first800labels = head(rt.label,800)

#running pca on the training data
pca <- prcomp(first800stories)

#looking at the first principal component(eignevector)
pc1 = pca$rotation[,1]
pc1sorted = sort(pc1)
#looking at 30 words with the most negative projections from the
#first component
head(pc1sorted,30)
#looking at 30 words with the most positive projections from the
#first component
tail(pc1sorted,30)

#determining how many principal components we need in order to 
#explain 80% of the variance of the data
variance = pca$sdev^2
sum(variance)
partialsum <- array(1:800)
for (i in 1:800){ partialsum[i] = sum(variance[1:i])}
partialsum = partialsum/(sum(variance))
N <- array(1:800)
plot(N,partialsum)
for (i in 1:800){
  if (partialsum[i] > 0.8)
  {
    number = i
    break
  }
}
number

##running the actual model 
rt.bow.training <- rt.bow[1:800,]
rt.label.training <- rt.label[1:800]
rt.bow.testing <- rt.bow[801:2000,]
rt.label.testing <- rt.label[801:2000]
rt.pca = prcomp(rt.bow.training)


#compute the loadings on first and second principal components
numPCA <- 2
matrixX.training <- rt.bow.training %*% rt.pca$rotation[,1:numPCA]
#use loadings as explanatory variables and labels as response variable
#to run a linear regression
matrixY.training <- rt.label.training
model <- lm(matrixY.training ˜ matrixX.training)

#calculating in-sample accuracy
accuracy.training <- sum((fitted(model) > 0.5) == rt.label.training) /
  length(rt.label.training)
accuracy.training

#computing the loadings of the testing data with the principal 
#components produced by training data
matrixX.testing <- rt.bow.testing %*% rt.pca$rotation[,1:numPCA]
matrixX.testing <- cbind(rep(1, 1200), matrixX.testing)

#using the coefficients from the training model to calculate the 
#out of sample accuracy 
matrixY.testing <- matrixX.testing %*% model$coefficients
prediction <- matrixY.testing > 0.5
accuracy.test <- sum((prediction == rt.label.testing)) / length(rt.label.testing)
accuracy.test

#increasing number of components to see if in-sample and out-of-sample
#accuracy increase or not
numPCAList <- 1:20
inSampleAccu <- c()
outSampleAccu <- c()
for (numPCA in numPCAList) {
  matrixX.training <- rt.bow.training %*% rt.pca$rotation[,1:numPCA]
  matrixY.training <- rt.label.training
  model <- lm(matrixY.training ˜ matrixX.training)
  # in-sample regression
  inSampleAccu[numPCA] <- sum((fitted(model) > 0.5) == rt.label.training) /
    length(rt.label.training)
  # out-of-sample regression
  rt.bow.testing <- rt.bow[801:2000,]
  rt.label.testing <- rt.label[801:2000]
  matrixX.testing <- rt.bow.testing %*% rt.pca$rotation[,1:numPCA]
  matrixX.testing <- cbind(rep(1, 1200), matrixX.testing)
  matrixY.testing <- matrixX.testing %*% model$coefficients
  prediction <- matrixY.testing > 0.5
  outSampleAccu[numPCA] <- sum((prediction == rt.label.testing)) /
    length(rt.label.testing)
}


plot(numPCAList, inSampleAccu, type = ’n’, xlab = ’number of PCs’, ylab = ’accuracy’)
lines(numPCAList, inSampleAccu, col = ’red’, lwd = 2.5)
lines(numPCAList, outSampleAccu, col = ’blue’, lwd = 2.5)
legend("bottomright", c("in-sample", "out-of-sample"), col = c(’red’,’blue’), lwd =
         c(2.5,2.5))


#training a neural network instead 
x_train <- rt.bow[1:800,]
y_train <- rt.label[1:800]
x_test <- rt.bow[801:2000,]
y_test <- rt.label[801:2000]
num_classes <- 2
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)
max_words <- 1000
batch_size <- 32
epochs <- 20

model <- keras_model_sequential()
model %>%  
  layer_dense(units = 512, input_shape = c(max_words)) %>%
  layer_activation(activation = 'relu') %>%  
  layer_dropout(rate = 0.5) %>%  
  layer_dense(units = num_classes) %>%  
  layer_activation(activation = 'softmax')
model %>% compile(  
  loss = 'categorical_crossentropy',  
  optimizer='sgd',  
  metrics = c('accuracy'))
history <- model %>% fit(
  x_train, y_train,   
  epochs = epochs, 
  batch_size = batch_size,   
  verbose = 1,  
  validation_split = 0.1)
score <- model %>% evaluate(x_test, y_test)
score[[2]]


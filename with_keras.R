# https://keras.rstudio.com/

# Here i try to replicate with keras some of the examples in dnnUtilsTests.R
# with dnnUtils.R functions

library(readr)
train_x <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/train_x.csv", 
                    col_names = FALSE)
train_y <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/train_y.csv", 
                    col_names = FALSE)
test_x  <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/test_x.csv", 
                    col_names = FALSE)
test_y  <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/test_y.csv", 
                    col_names = FALSE)

train_x <- t(as.matrix(train_x)) 
train_y <- t(as.matrix(train_y)) 
test_x  <- t(as.matrix(test_x)) 
test_y  <- t(as.matrix(test_y))

library(keras)
# train_y <- to_categorical(train_y)
# test_y  <- to_categorical(test_y)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(12288)) %>% 
  # layer_dropout(rate = 0.4) %>% 
  layer_dense(units =  7, activation = 'relu') %>%
  # layer_dropout(rate = 0.3) %>%
  layer_dense(units =  5, activation = 'relu') %>%
  # layer_dropout(rate = 0.3) %>%
  layer_dense(units =  1, activation = 'sigmoid')

summary(model)

model %>% compile(
  # loss      = 'categorical_crossentropy',
  # optimizer = optimizer_rmsprop(),
  loss      = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.0075),
  metrics   = c('accuracy', 'binary_accuracy')
)

history <- model %>% fit(train_x, train_y, 
                         epochs           = 2500, 
                         batch_size       = 20, 
                         validation_split = 0.2
                         )

plot(history)

model %>% evaluate(test_x, test_y)

pred <- model %>% predict_classes(test_x)

sum(pred == test_y) / nrow(test_y)

probs <- model %>% predict_proba(test_x)
probs

#########
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 40, activation = 'relu', input_shape = c(12288)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 4, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

summary(model)

model %>% compile(
  # loss = 'categorical_crossentropy',
  # optimizer = optimizer_rmsprop(),
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(lr = 0.0075),
  metrics = c('accuracy')
)

history <- model %>% fit(
  train_x, train_y, 
  epochs = 100, batch_size = 209, 
  validation_split = 0
)

plot(history)

model %>% evaluate(test_x, test_y, batch_size = 50)

pred <- model %>% predict_classes(test_x)

sum(pred == test_y) / nrow(test_y)

probs <- model %>% predict_proba(test_x)
probs

###### ADAM
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(12288)) %>% 
  # layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 7, activation = 'relu') %>%
  # layer_dropout(rate = 0.3) %>%
  layer_dense(units = 5, activation = 'relu') %>%
  # layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

summary(model)

model %>% compile(
  # loss = 'categorical_crossentropy',
  # optimizer = optimizer_rmsprop(),
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(lr = 0.001),
  metrics = c('accuracy', 'binary_accuracy')
)

history <- model %>% fit(
  train_x, train_y, 
  epochs = 300, batch_size = 32, 
  validation_split = 0.2
)

plot(history)

model %>% evaluate(test_x, test_y)

pred <- model %>% predict_classes(test_x)

sum(pred == test_y) / nrow(test_y)

probs <- model %>% predict_proba(test_x)
probs


# Sign language images ----------------------------------------------------

library(readr)
train_x <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/X_train_norm.csv", 
                    col_names = FALSE)
train_y <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/Y_train.csv", 
                    col_names = FALSE)
test_x  <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/X_test_norm.csv", 
                    col_names = FALSE)
test_y  <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/Y_test.csv", 
                    col_names = FALSE)

train_x <- t(as.matrix(train_x)) 
train_y <- t(as.matrix(train_y)) 
test_x  <- t(as.matrix(test_x)) 
test_y  <- t(as.matrix(test_y))

library(keras)
train_y <- to_categorical(train_y)
test_y  <- to_categorical(test_y)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 25, activation = 'relu', input_shape = ncol(train_x)) %>% 
  # layer_dropout(rate = 0.6) %>% 
  layer_dense(units = 12, activation = 'relu') %>%
  # layer_dropout(rate = 0.3) %>%
  layer_dense(units =  6, activation = 'relu') %>%
  # layer_dropout(rate = 0.3) %>%
  layer_dense(units =  6, activation = 'softmax')

summary(model)

model %>% compile(loss      = 'categorical_crossentropy',
                  optimizer = optimizer_adam(lr    = 0.0001),
                  metrics   = c('accuracy')
                  )

set.seed(1)

history <- model %>% fit(train_x, train_y, 
                         epochs           = 1500, 
                         batch_size       = 32, 
                         validation_split = 0
)

plot(history)

model %>% evaluate(test_x, test_y)

pred <- model %>% predict_classes(test_x)

orig_test_y <- apply(test_y, 1, function(x) which(x == 1)) - 1
sum(pred == orig_test_y) / nrow(test_y)

probs <- model %>% predict_proba(test_x)
probs

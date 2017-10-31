library(tensorflow)

one_hot_matrix <- function(labels, C) {
  "
  Creates a matrix where the i-th row corresponds to the ith class number and the jth column
  corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
  will be 1. 
  
  Arguments:
  labels -- vector containing the labels 
  C -- number of classes, the depth of the one hot dimension
  
  Returns: 
  one_hot -- one hot matrix
  "
  
  C      <- tf$constant(as.integer(C), name = "C", dtype = tf$int32)
  
  labels <- tf$constant(as.integer(labels), name ="labels", dtype = tf$int32)
  
  on_hot_matrix <- tf$one_hot(indices = labels, depth = C, axis = 0)
  
  sess <- tf$Session()
  
  one_hot <- sess$run(on_hot_matrix)
  
  sess$close()
  
  return(one_hot)
}


# SIGNS Dataset -----------------------------------------------------------


library(readr)
train_x <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/X_train_norm.csv", 
                    col_names = FALSE)
train_y <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/Y_train.csv", 
                    col_names = FALSE)
test_x  <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/X_test_norm.csv", 
                    col_names = FALSE)
test_y  <- read_csv("D:/PROYECTOS DATA SCIENCE/DNN/data/Y_test.csv", 
                    col_names = FALSE)

train_x_m <- as.matrix(train_x)
train_y_m <- one_hot_matrix(t(as.matrix(train_y)), 6)
test_x_m  <- as.matrix(test_x) 
test_y_m  <- one_hot_matrix(t(as.matrix(test_y)), 6)

dim(train_x_m)
dim(train_y_m)
dim(test_x_m)
dim(test_y_m)

index <- 1
A <- 64 * 64
flattened_im <- train_x_m[,index]
im_seq <- seq(1, 3*A-2, by = 3)
im_r <- matrix(flattened_im[im_seq],   nrow = 64, ncol = 64)
im_g <- matrix(flattened_im[im_seq+1], nrow = 64, ncol = 64)
im_b <- matrix(flattened_im[im_seq+2], nrow = 64, ncol = 64)
im_r <- t(im_r)
im_g <- t(im_g)
im_b <- t(im_b)

col <- rgb(im_r, im_g, im_b)
dim(col) <- dim(im_r)

library(grid)
grid.raster(col, interpolate=FALSE)


# Create placeholders -----------------------------------------------------

create_placeholders <- function(n_x, n_y) {
  "
Creates the placeholders for the tensorflow session.
    
  Arguments:
  n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
  n_y -- scalar, number of classes (from 0 to 5, so -> 6)
  
  Returns:
  X -- placeholder for the data input, of shape [n_x, None] and dtype 'float'
  Y -- placeholder for the input labels, of shape [n_y, None] and dtype 'float'
  
  Tips:
  - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
  In fact, the number of examples during test/train is different.
  "
  X <- tf$placeholder(dtype = tf$float32, shape = shape(n_x, NULL), name = "X")
  Y <- tf$placeholder(dtype = tf$float32, shape = shape(n_y, NULL), name = "Y")
  
  return(list(X = X, Y = Y))
}

# TEST IT
ph_list <- create_placeholders(64*64*3, 6)
ph_list


# Initializing the parameters ---------------------------------------------

initialize_parameters <- function(seed = 1) {
  "
  Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
  b1 : [25, 1]
  W2 : [12, 25]
  b2 : [12, 1]
  W3 : [6, 12]
  b3 : [6, 1]
  
  Returns:
  parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
  "
  
  tf$set_random_seed(seed)
  
  W1 <- tf$get_variable("W1", c(25, 12288), 
                        initializer = tf$contrib$layers$xavier_initializer(seed = seed))
  b1 <- tf$get_variable("b1", c(25, 1), initializer = tf$zeros_initializer())
  W2 <- tf$get_variable("W2", c(12, 25), 
                        initializer = tf$contrib$layers$xavier_initializer(seed = seed))
  b2 <- tf$get_variable("b2", c(12, 1), initializer = tf$zeros_initializer())
  W3 <- tf$get_variable("W3", c(6, 12), 
                        initializer = tf$contrib$layers$xavier_initializer(seed = seed))
  b3 <- tf$get_variable("b3", c(6, 1), initializer = tf$zeros_initializer())
  out <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2, W3 = W3, b3 = b3)

  return(out)
}

# TEST IT
tf$reset_default_graph()
sess <-  tf$Session()

with(sess, {
  parameters <- initialize_parameters()
  print(parameters)
})
sess$close()


# Forward propagation in tensorflow ---------------------------------------

forward_propagation <- function(X, parameters) {
  '
   Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2, b2, W3, b3
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
  '
  
  W1 <- parameters[["W1"]]
  b1 <- parameters[["b1"]]
  W2 <- parameters[["W2"]]
  b2 <- parameters[["b2"]]
  W3 <- parameters[["W3"]]
  b3 <- parameters[["b3"]]
  
  Z1 <- tf$add(tf$matmul(W1,  X), b1)
  A1 <- tf$nn$relu(Z1)
  Z2 <- tf$add(tf$matmul(W2, A1), b2)
  A2 <- tf$nn$relu(Z2)
  Z3 <- tf$add(tf$matmul(W3, A2), b3)
  A3 <- tf$nn$relu(Z3)
  
  return(Z3)
}

# TEST IT
tf$reset_default_graph()

with(tf$Session() %as% sess, {
  ph_list <- create_placeholders(12288, 6)
  parameters <- initialize_parameters()
  Z3 <- forward_propagation(ph_list$X, parameters)
  print(Z3)
})


# Compute cost ------------------------------------------------------------

compute_cost <- function(Z3, Y) {
  "
  Computes the cost
  
  Arguments:
  Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
  Y -- 'true' labels vector placeholder, same shape as Z3
  
  Returns:
  cost - Tensor of the cost function
  "
  logits <- tf$transpose(Z3)
  labels <- tf$transpose(Y)
  
  cost <- tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits = logits, 
                                                                 labels = labels))

  return(cost)
}

# TRY IT
tf$reset_default_graph()

with(tf$Session() %as% sess, {
  ph_list <- create_placeholders(12288, 6)
  parameters <- initialize_parameters()
  Z3 <- forward_propagation(ph_list$X, parameters)
  cost <- compute_cost(Z3, ph_list$Y)
  print(cost)
})


# Random mini batches -----------------------------------------------------

random_mini_batches <- function(X, Y, mini_batch_size = 64, seed = 0) {
  "
  Creates a list of random minibatches from (X, Y)
  
  Arguments:
  X -- input data, of shape (input size, number of examples)
  Y -- true 'label' vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
  mini_batch_size - size of the mini-batches, integer
  seed -- this is only for the purpose of grading, so that your random minibatches are the same as ours.
  
  Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
  "
  m <- ncol(X)
  mini_batches <- list()
  
  set.seed(seed)
  
  # Step 1: Shuffle (X, Y)
  permutation <- sample(x = 1:m, size = m, replace = FALSE)
  shuffled_X <- X[, permutation]
  shuffled_Y <- Y[, permutation]
  
  # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
  num_complete_minibatches <- floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
  for(k in 1:num_complete_minibatches) {
    idxs <- ((k-1) * mini_batch_size + 1) : ((k-1) * mini_batch_size + mini_batch_size)
    # print(idxs)
    mini_batch_X <- shuffled_X[, idxs]
    mini_batch_Y <- shuffled_Y[, idxs]
    mini_batch   <- list(mini_batch_X, mini_batch_Y)
    mini_batches[[k]] <- mini_batch
  }
  
  # Handling the end case (last mini-batch < mini_batch_size)
  if (m %% mini_batch_size != 0) {
    idxs <- (num_complete_minibatches * mini_batch_size + 1) : m
    # print(idxs)
    mini_batch_X <- shuffled_X[, idxs]
    mini_batch_Y <- shuffled_Y[, idxs]
    mini_batch   <- list(mini_batch_X, mini_batch_Y)
    mini_batches[[k+1]] <- mini_batch
  }
  
  return(mini_batches)
}

# TEST IT
random_mini_batches(train_x_m[, 1:12], train_y_m[, 1:12], 
                    mini_batch_size = 4, seed = 0)

# Building the model ------------------------------------------------------

model <- function(X_train, Y_train, 
                  X_test, Y_test, 
                  learning_rate = 0.0001,
                  num_epochs = 1500, 
                  minibatch_size = 32, 
                  print_cost = TRUE, 
                  seed = 1) {
"
Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

Arguments:
X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
X_test -- training set, of shape (input size = 12288, number of training examples = 120)
Y_test -- test set, of shape (output size = 6, number of test examples = 120)
learning_rate -- learning rate of the optimization
num_epochs -- number of epochs of the optimization loop
minibatch_size -- size of a minibatch
print_cost -- True to print the cost every 100 epochs

Returns:
parameters -- parameters learnt by the model. They can then be used to predict.
"
  tf$reset_default_graph()      # to be able to rerun the model without overwriting tf variables
  tf$set_random_seed(seed)
  seed <- seed + 2
  n_x <- nrow(X_train)  # (n_x: input size, m : number of examples in the train set)
  m   <- ncol(X_train)
  n_y <- dim(Y_train)[1]         # n_y : output size
  costs <- c()                # To keep track of the cost
  
  # Create Placeholders of shape (n_x, n_y)
  ph_list <- create_placeholders(n_x, n_y)
  X <- ph_list[["X"]]
  Y <- ph_list[["Y"]]

  # Initialize parameters
  parameters <- initialize_parameters()
  
  # Forward propagation: Build the forward propagation in the tensorflow graph
  Z3 <- forward_propagation(X, parameters)
  
  # Cost function: Add cost function to tensorflow graph
  cost <- compute_cost(Z3, Y)
  
  # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
  optimizer <- tf$train$AdamOptimizer(learning_rate = learning_rate)$minimize(cost)
  
  # Initialize all the variables
  init <- tf$global_variables_initializer()
  
  # Start the session to compute the tensorflow graph
  with(tf$Session() %as% sess, {
    
    # Run the initialization
    sess$run(init)
    
    # Do the training loop
    for (epoch in 1:num_epochs) {
      epoch_cost <- 0 # Defines a cost related to an epoch
      
      seed <- seed + 1
      
      minibatches <- random_mini_batches(X_train, Y_train, 
                                         minibatch_size, seed)
      num_minibatches <- length(minibatches) # number of minibatches of size minibatch_size in the train set
      
      
      for(mbtch in minibatches) {
        # IMPORTANT: The line that runs the graph on a minibatch.
        # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
        opt_res <- sess$run(list(optimizer, cost),
                            feed_dict = dict(X = mbtch[[1]],
                                             Y = mbtch[[2]]))
        
        epoch_cost <- epoch_cost + opt_res[[2]] / num_minibatches
      }
      
      # Print the cost every epoch
      if(print_cost == TRUE && epoch %% 100 == 0) {
        print(sprintf("Cost after epoch %d: %f", epoch, epoch_cost))
      }
      if(print_cost == TRUE && epoch %% 5 == 0) {
        costs <- c(costs, epoch_cost)
      }
    }
    
    # plot the cost
    plot(1:length(costs), costs,
         type = "l", 
         xlab = "Iterations (per tens)",
         main = sprintf("Learning rate = %f", learning_rate))
    
    # lets save the parameters in a variable
    parameters <- sess$run(parameters)
    print ("Parameters have been trained!")
    
    # Calculate the correct predictions
    correct_prediction = tf$equal(tf$argmax(Z3), tf$argmax(Y))
    
    # Calculate accuracy on the test set
    accuracy = tf$reduce_mean(tf$cast(correct_prediction, "float"))
    
    print (sprintf("Train Accuracy: %f", 
                   accuracy$eval(dict(X = X_train, Y = Y_train))))
    print (sprintf("Test Accuracy: %f",
                   accuracy$eval(dict(X = X_test, Y = Y_test))))
  })
  
  return(parameters)
}

# TEST IT
parameters = model(train_x_m, train_y_m, test_x_m, test_y_m)


# Predict -----------------------------------------------------------------

predict <- function(X, parameters) {
  
  W1 <- tf$convert_to_tensor(parameters["W1"])
  b1 <- tf$convert_to_tensor(parameters["b1"])
  W2 <- tf$convert_to_tensor(parameters["W2"])
  b2 <- tf$convert_to_tensor(parameters["b2"])
  W3 <- tf$convert_to_tensor(parameters["W3"])
  b3 <- tf$convert_to_tensor(parameters["b3"])
  
  params <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2, W3 =  W3, b3 = b3)
  
  x <- tf$placeholder("float", c(12288, 1))
  
  z3 <- forward_propagation(x, params)
  p <- tf$argmax(z3)
  
  sess <- tf$Session()
  prediction = sess$run(p, feed_dict = dict(x = X))
  
  return(prediction)
}

# TEST IT



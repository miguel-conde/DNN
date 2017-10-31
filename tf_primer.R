## Tensorflow tutorial


# Tensorflow Basics -------------------------------------------------------


library(tensorflow)

# Create tensors (variables / constants)
y_hat <- tf$constant(36, name = "y_hat")
y     <- tf$constant(39, name = "y")

# Create operations with those tensors
loss <- tf$Variable((y - y_hat)^2, name = "loss")

# Initialize Tensors
init <- tf$global_variables_initializer()

# Crete a session
sess <- tf$Session()

# Run
sess$run(init)        # Initialization
print(sess$run(loss)) # Operations

# Pay attention to this example:
a <- tf$constant(2)
b <- tf$constant(10)
c <- tf$multiply(a, b)
print(c)

# In order to actually multiply the two numbers, you have to create a 
# session and run it.
sess$run(c)


# Placeholders ------------------------------------------------------------

# A placeholder is an object whose value you can specify only later. To 
# specify values for a placeholder, you can pass in values by using a "feed 
# dictionary" (feed_dict variable). Below, we created a placeholder for x. 
# This allows us to pass in a number later when we run the session.
x  <- tf$placeholder(tf$int64, name = "x")
op <- tf$multiply(tf$constant(2L, dtype = tf$int64), x)

print(sess$run(op, 
               feed_dict = dict(x = 3)))

# When you specify the operations needed for a computation, you are telling 
# TensorFlow how to construct a computation graph. The computation graph can 
# have some placeholders whose values you will specify only later. Finally, when 
# you run the session, you are telling TensorFlow to execute the computation 
# graph.

# When finished, don't forget to close the session
sess$close()

# Linear function ---------------------------------------------------------

linear_function <- function(seed = 1) {
"
Implements a linear function: 
  Initializes W to be a random tensor of shape (4,3)
  Initializes X to be a random tensor of shape (3,1)
  Initializes b to be a random tensor of shape (4,1)
  Returns: 
  result -- runs the session for Y = WX + b 
"
  set.seed(seed = seed)
  
  X <- tf$constant(runif(3*1), shape = c(3, 1), name = "X")
  W <- tf$constant(runif(4*3), shape = c(4, 3), name = "X")
  b <- tf$constant(runif(4*1), shape = c(4, 1), name = "X")
  
  Y <- tf$add(tf$matmul(W, X), b)
  
  sess <- tf$Session()
  res  <- sess$run(Y)
  
  sess$close()
  
  return(res)
}

# TEST IT
print(linear_function())



# Sigmoid -----------------------------------------------------------------

sigmoid <- function(z) {
"
Computes the sigmoid of z
    
  Arguments:
  z -- input value, scalar or vector
  
  Returns: 
  results -- the sigmoid of z
"
  
  x <- tf$placeholder(tf$float32, name = "x")
  
  sigmoid <- tf$sigmoid(x)
  
  sess <- tf$Session()
  res  <- sess$run(sigmoid, feed_dict = dict(x = z))
  sess$close()
  
  return(res)
}

# TEST IT
print(sprintf("sigmoid(0) = %f", sigmoid(0)))
print(sprintf("sigmoid(12) = %f", sigmoid(12)))

x <- seq(-20, 20, by = 0.1)
y <- sigmoid(x)
plot(x, y, type = "l", main = "Sigmoid", ylab ="Sigmoid(x)")


# Cost function -----------------------------------------------------------

cost <- function(logits, labels) {
"
Computes the cost using the sigmoid cross entropy

  Arguments:
  logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
  labels -- vector of labels y (1 or 0) 
  
  Note: What we ve been calling 'z' and 'y' in this class are respectively called 'logits' and 'labels' 
  in the TensorFlow documentation. So logits will feed into z, and labels into y. 
  
  Returns:
  cost -- runs the session of the cost (formula (2))
"
  z <- tf$placeholder(tf$float32, name = "z") # logits
  y <- tf$placeholder(tf$float32, name = "y") # labels
  
  cost <- tf$nn$sigmoid_cross_entropy_with_logits(logits = z, labels = y)
  
  sess <- tf$Session()
  
  cost <- sess$run(cost, feed_dict = dict(z = logits, y = labels))
  
  sess$close()
  
  return(cost)
}

# TEST IT
logits <- sigmoid(c(0.2,0.4,0.7,0.9))
cost   <- cost(logits, c(0, 0, 1, 1))
print(cost)


# One Hot Encodings -------------------------------------------------------

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

# TEST IT
labels <- c(1,2,3,0,2,1)
one_hot <- one_hot_matrix(labels, C = 4)
print(one_hot)


# Initialize with zeros and ones ------------------------------------------

ones <- function(shape) {
"
  Creates an array of ones of dimension shape
    
  Arguments:
  shape -- shape of the array you want to create
  
  Returns: 
  ones -- array containing only ones
"
  ones <- tf$ones(shape)
  
  sess <- tf$Session()
  
  ones <- sess$run(ones)
  
  sess$close()
  
  return(ones)
}

# TEST IT
print(ones(c(3)))
print(ones(c(3,4)))

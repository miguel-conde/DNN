log_rnd_unif <- function(n, min, max, seed = 1) {
  set.seed(seed)
  lrs <- 10^runif(n = n, min = log10(min), max = log10(max))
  lrs[order(lrs)]
}

fit_mod_SGD_lr <- function(l_r, n_epochs, n_x,  
                           x_train, y_train, 
                           x_dev, y_dev, 
                           batch_size,
                           verbose = 1, seed = 1) {

  set.seed(seed)
  
  # DEFINE the Neural Network model
  model_i <- keras_model_sequential() 
  model_i %>% 
    layer_dense(units = 256, activation = 'relu', input_shape = c(n_x)) %>% 
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')
  
  # COMPILE the model
  model_i %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_sgd(lr       = l_r, 
                              momentum = 0.0, 
                              decay    = 0.0, 
                              nesterov = FALSE),
    metrics = c('accuracy')
  )
  
  # FIT the model
  fit_time <- system.time(
    history <- model_i %>% fit(
      x_train, y_train, 
      verbose = verbose, 
      epochs           = n_epochs, 
      batch_size       = batch_size, 
      validation_data  = list(x_dev, y_dev)
    )
  )
  
  return(list(model_i = model_i, history = history, fit_time = fit_time))
}

tune_lr <- function(lrs, n_epochs, n_x,  
                    x_train, y_train, 
                    x_dev, y_dev, 
                    batch_size,
                    verbose = 1, seed = 1) {
  model_best <- NULL
  acc_best <- 0
  perfs <- list()
  
  for(i in 1:length(lrs)) {
    
    ### CHANGE THIS TO TRY OTHER ALTERNATIVES
    res <- fit_mod_SGD_lr(lrs[i], n_epochs, n_x,  
                          x_train, y_train, 
                          x_dev, y_dev, 
                          batch_size,
                          verbose = verbose, seed = seed)
    ###
    
    res_i <- list(learning_rate = lrs[i],
                  fit_time      = res$fit_time,
                  history       = res$history)
    perfs[[i]] <- res_i
    
    if (acc_best < res_i$history$metrics$val_acc[n_epochs]) {
      model_best <- res$model_i
      acc_best <- res_i$history$metrics$val_acc[n_epochs]
    }
  } # END FOR
  
  return(list(best_model = model_best, performances_list = perfs))
}

summ_lr <- function(m_lr_fit) {
  m_lr_df <- sapply(m_lr_fit$performances_list, function(x) {
    n_epochs <- x$history$params$epochs
    c(learning_rate = x$learning_rate, 
      train_acc = x$history$metrics$acc[n_epochs], 
      val_acc   = x$history$metrics$val_acc[n_epochs])
  })
  out <- as.data.frame(t(m_lr_df))
  # out[order(out$val_acc),]
  out
}

plot_lr <- function(m_lr_fit, bayes_acc = 1) {
  plot_data <- summ_lr(m_lr_fit)
  ymax<- max(plot_data$val_acc, plot_data$train_acc)
  ymin<- min(plot_data$val_acc, plot_data$train_acc)
  plot(x = plot_data$learning_rate, y = plot_data$val_acc,
       ylim = c(0.9995*ymin, 1.0005*ymax),
       type = "o", col = "red",
       xlab = "Learning Rate",
       ylab = "Accuracy",
       main = "Accuracy vs. Learning Rate")
  lines(x = plot_data$learning_rate, y = plot_data$train_acc,
        type = "l", col = "blue")
  abline(a = bayes_acc, b = 0,
         lty = 2, col = "black")
  legend("bottomright", 
         legend = c("Dev Test", "Training Test", "Bayes"), 
         col=c("red", "blue", "black"), lty = c(1, 1, 2), pch = c(1, NA, NA)) 
}

m1_res <- fit_mod_SGD_lr(l_r = 0.01, n_epochs = 30, n_x,  
                         x_train, y_train, 
                         x_dev = x_test, y_dev = y_test, 
                         batch_size = m)
m1_res$fit_time
plot(m1_res$history)

set.seed(1)
lrs = 10^runif(n = 3, min = log10(0.001), max = log10(0.1))
m1_lr_fit <- tune_lr(lrs = lrs, 
                     n_epochs = 30, n_x,  
                     x_train, y_train, 
                     x_dev = x_test, y_dev = y_test, 
                     batch_size = m)

m1_lr_fit$performances_list[[1]]
str(m1_lr_fit$performances_list[[1]]$history)
plot(m1_lr_fit$performances_list[[1]]$history)



summ_lr(m1_lr_fit)


m1_lr_fit <- tune_lr(lrs = lrs, 
                     n_epochs = 30, n_x,  
                     x_train, y_train, 
                     x_dev = x_test, y_dev = y_test, 
                     batch_size = 256)

summ_lr(m1_lr_fit)
plot(m1_lr_fit$performances_list[[3]]$history)


#### 



lrs <- log_rnd_unif(n = 20, min = 0.001, max = 1)
m1_lr_fit <- tune_lr(lrs = lrs, 
                     n_epochs = 30, n_x,  
                     x_train, y_train, 
                     x_dev = x_test, y_dev = y_test, 
                     batch_size = 256)

summ_lr(m1_lr_fit)
plot_lr(m1_lr_fit, bayes_acc = (1-0.23/100))
plot(m1_lr_fit$performances_list[[19]]$history)


rnd_nu <- function() {
  sample(1:512, 1)
}

rnd_NN_layers <- function(n_str = 10, max_hl = 2, nL = 1, seed = 1) {
  "
Generates a random structure for a NN

Arguments:
  - n_str: number of structures to generate
  - max_hl: maximum number of hidden layers
  - nL: number of units in the output layer

Return:
  - A list whose elements are integer vectors specifying the NN structure.
"
  set.seed(seed)
  
  list_n <- list() # lista para retornar
  
  for(i in 1:n_str) { # FOR_1 : para cada NN struct i

    n_units_l <- c() # Vector con los numeros de units en cada layer
    
    # ¿Cuantas layers va a tener?
    n_layers = sample(1:max_hl, 1)
    
    # ¿Cuantas units va a tener cada layer de esta NN
    n_units_l <- c(2^sample(1:10, n_layers), 1)
    
    # Añadir a la lista la especificacion de esta NN struct i 
    list_n[[i]] <- n_units_l

  } # END FOR_1: NN struct
  
  return(list_n)
}

rnd_NN_layers()
rnd_NN_layers(n_str = 5, max_hl = 10)
rnd_NN_layers(n_str = 10, max_hl = 5)

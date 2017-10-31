pacman::p_load(imager)


plot(boats)

class(boats)
str(boats)
length(boats)
boats
dim(boats)
256*384*3
summary(boats)

grayscale(boats)

my_image <- "C:/Users/Miguel/OneDrive/CURSOS/Coursera/Deep Learning Spec/1 - NNs and Deep Learning/W4 - Deep NN/C - PROGRAMMING ASSIGNMENTS/b - DNN - Application/images/IMG_1629.JPG"
my_cat <- load.image(my_image)
plot(my_cat)

my_cat
dim(my_cat)
length(my_cat)

my_flattened_cat <- as.numeric(my_cat)
str(my_flattened_cat)


####
my_image <- "C:/Users/Miguel/OneDrive/CURSOS/Coursera/Deep Learning Spec/1 - NNs and Deep Learning/W4 - Deep NN/C - PROGRAMMING ASSIGNMENTS/b - DNN - Application/images/my_image.jpg"
a_cat <- load.image(my_image)
plot(a_cat)

a_cat
dim(a_cat)
length(a_cat)

a_cat2 <- resize(a_cat, 64, 64)
plot(a_cat2)

a_flattened_cat <- as.numeric(a_cat)
str(a_flattened_cat)

A <- dim(a_cat)[1] * dim(a_cat)[2]
a_cat_r <- matrix(a_flattened_cat[1:A], 
                  nrow = dim(a_cat)[1], ncol = dim(a_cat)[2])
a_cat_g <- matrix(a_flattened_cat[(A+1):(2*A)], 
                  nrow = dim(a_cat)[1], ncol = dim(a_cat)[2])
a_cat_b <- matrix(a_flattened_cat[(2*A+1):(3*A)], 
                  nrow = dim(a_cat)[1], ncol = dim(a_cat)[2])
a_cat_r <- t(a_cat_r)
a_cat_g <- t(a_cat_g)
a_cat_b <- t(a_cat_b)

col <- rgb(a_cat_r, a_cat_g, a_cat_b)
dim(col) <- dim(a_cat_r)

library(grid)
grid.raster(col, interpolate=FALSE)




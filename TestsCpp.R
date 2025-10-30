
# Header for Rcpp and RcppArmadillo
library(Rcpp)
library(RcppArmadillo)
library(microbenchmark)

# Source your C++ funcitons
sourceCpp("LassoInC.cpp")

# Source your LASSO functions from HW4 (make sure to move the corresponding .R file in the current project folder)
source("LassoFunctions.R")

# Do at least 2 tests for soft-thresholding function below. You are checking output agreements on at least 2 separate inputs
#################################################

# Test case 1: Positive value
a1 <- 5.0
lambda1 <- 2.0
r_result1 <- soft(a1, lambda1)
cpp_result1 <- soft_c(a1, lambda1)
cat("Test 1 - Positive: R =", r_result1, ", C++ =", cpp_result1, 
    ", Match:", all.equal(r_result1, cpp_result1), "\n")
# Test 1 - Positive: R = 3 , C++ = 3 , Match: TRUE 

# Test case 2: Negative value
a2 <- -5.0
lambda2 <- 2.0
r_result2 <- soft(a2, lambda2)
cpp_result2 <- soft_c(a2, lambda2)
cat("Test 2 - Negative: R =", r_result2, ", C++ =", cpp_result2, 
    ", Match:", all.equal(r_result2, cpp_result2), "\n")
# Test 2 - Negative: R = -3 , C++ = -3 , Match: TRUE 

# Test case 3: Value within threshold (should be 0)
a3 <- 1.5
lambda3 <- 2.0
r_result3 <- soft(a3, lambda3)
cpp_result3 <- soft_c(a3, lambda3)
cat("Test 3 - Within threshold: R =", r_result3, ", C++ =", cpp_result3, 
    ", Match:", all.equal(r_result3, cpp_result3), "\n")
# Test 3 - Within threshold: R = 0 , C++ = 0 , Match: TRUE 


# Do at least 2 tests for lasso objective function below. You are checking output agreements on at least 2 separate inputs
#################################################
# Generate simulation data data
n <- 100
p <- 10
X_raw <- matrix(rnorm(n * p), n, p)
Y_raw <- rnorm(n)

# Center Y, # Center and scale X
Ymean <- mean(Y_raw)
Ytilde <- Y_raw - Ymean
Xmeans <- colMeans(X_raw)
X_centered <- sweep(X_raw, 2, Xmeans, FUN = "-")
weights <- sqrt(colSums(X_centered^2) / n)
Xtilde <- sweep(X_centered, 2, weights, FUN = "/")

# Test case 1: Random beta
beta1 <- rnorm(p)
lambda_test1 <- 0.5
r_obj1 <- lasso(Xtilde, Ytilde, beta1, lambda_test1)
cpp_obj1 <- lasso_c(Xtilde, Ytilde, beta1, lambda_test1)
cat("Test 1 - Random beta: R =", r_obj1, ", C++ =", cpp_obj1, 
    ", Match:", all.equal(r_obj1, cpp_obj1), "\n")
# Test 1 - Random beta: R = 5.71375 , C++ = 5.71375 , Match: TRUE 

# Test case 2: Zero beta
beta2 <- rep(0, p)
lambda_test2 <- 1.0
r_obj2 <- lasso(Xtilde, Ytilde, beta2, lambda_test2)
cpp_obj2 <- lasso_c(Xtilde, Ytilde, beta2, lambda_test2)
cat("Test 2 - Zero beta: R =", r_obj2, ", C++ =", cpp_obj2, 
    ", Match:", all.equal(r_obj2, cpp_obj2), "\n")
# Test 2 - Zero beta: R = 0.4326151 , C++ = 0.4326151 , Match: TRUE 

# Test case 3: Large lambda
beta3 <- rnorm(p, mean = 2, sd = 0.5)
lambda_test3 <- 10.0
r_obj3 <- lasso(Xtilde, Ytilde, beta3, lambda_test3)
cpp_obj3 <- lasso_c(Xtilde, Ytilde, beta3, lambda_test3)
cat("Test 3 - Large lambda: R =", r_obj3, ", C++ =", cpp_obj3, 
    ", Match:", all.equal(r_obj3, cpp_obj3), "\n")
# Test 3 - Large lambda: R = 222.7312 , C++ = 222.7312 , Match: TRUE 


# Do at least 2 tests for fitLASSOstandardized function below. You are checking output agreements on at least 2 separate inputs
#################################################
# Generate centered and scaled data
n <- 100
p <- 20
X_raw <- matrix(rnorm(n * p), n, p)
Y_raw <- rnorm(n)
Ymean <- mean(Y_raw)
Ytilde <- Y_raw - Ymean
Xmeans <- colMeans(X_raw)
X_centered <- sweep(X_raw, 2, Xmeans, FUN = "-")
weights <- sqrt(colSums(X_centered^2) / n)
Xtilde <- sweep(X_centered, 2, weights, FUN = "/")

# Test Case 1: lambda = 0.5, zero starting point
cat("Test 1 - lambda = 0.5, zero start:\n")
lambda1 <- 0.5
beta_start1 <- rep(0, p)
r_result1 <- fitLASSOstandardized(Xtilde, Ytilde, lambda1, beta_start1, eps = 0.001)
cpp_result1 <- fitLASSOstandardized_c(Xtilde, Ytilde, lambda1, beta_start1, eps = 0.001)
cat("R beta norm:", norm(as.matrix(r_result1$beta), "F"), "\n")
# R beta norm: 0  
cat("C++ beta norm:", norm(as.matrix(cpp_result1), "F"), "\n")
# C++ beta norm: 0 
cat("Beta match:", all.equal(as.vector(r_result1$beta), as.vector(cpp_result1)), "\n")
# Beta match: TRUE 
cat("Max difference:", max(abs(r_result1$beta - cpp_result1)), "\n\n")
# Max difference: 0


# Test Case 2: lambda = 0.1, random starting point
cat("Test 2 - lambda = 0.1, random start:\n")
lambda2 <- 0.1
beta_start2 <- rnorm(p, mean = 0, sd = 0.5)

r_result2 <- fitLASSOstandardized(Xtilde, Ytilde, lambda2, beta_start2, eps = 0.001)
cpp_result2 <- fitLASSOstandardized_c(Xtilde, Ytilde, lambda2, beta_start2, eps = 0.001)

cat("R beta norm:", norm(as.matrix(r_result2$beta), "F"), "\n")
# R beta norm: 0.05874695 
cat("C++ beta norm:", norm(as.matrix(cpp_result2), "F"), "\n")
# C++ beta norm: 0.05874695  
cat("Beta match:", all.equal(as.vector(r_result2$beta), as.vector(cpp_result2)), "\n")
# Beta match: TRUE  
cat("Max difference:", max(abs(r_result2$beta - cpp_result2)), "\n\n")
# Max difference: 4.163336e-17 

# Test Case 3: large lambda (should give sparse solution)
cat("Test 3 - lambda = 2.0 (sparse solution):\n")
lambda3 <- 2.0
beta_start3 <- rep(0, p)

r_result3 <- fitLASSOstandardized(Xtilde, Ytilde, lambda3, beta_start3, eps = 0.001)
cpp_result3 <- fitLASSOstandardized_c(Xtilde, Ytilde, lambda3, beta_start3, eps = 0.001)

cat("R non-zero coefficients:", sum(abs(r_result3$beta) > 1e-8), "\n")
# R non-zero coefficients: 0  
cat("C++ non-zero coefficients:", sum(abs(cpp_result3) > 1e-8), "\n")
# C++ non-zero coefficients: 0 
cat("Beta match:", all.equal(as.vector(r_result3$beta), as.vector(cpp_result3)), "\n")
# Beta match: TRUE 
cat("Max difference:", max(abs(r_result3$beta - cpp_result3)), "\n\n")
# Max difference: 0 

# Do microbenchmark on fitLASSOstandardized vs fitLASSOstandardized_c
######################################################################
lambda_bench <- 0.5
beta_start_bench <- rep(0, p)

mb1 <- microbenchmark(
  R = fitLASSOstandardized(Xtilde, Ytilde, lambda_bench, beta_start_bench, eps = 0.001),
  Cpp = fitLASSOstandardized_c(Xtilde, Ytilde, lambda_bench, beta_start_bench, eps = 0.001),
  times = 50
)
print(mb1)
#Unit: microseconds
#expr     min      lq       mean      median      uq     max      neval   cld
#R       107.215  118.039 124.76382 125.337     128.412   172.528    50  a 
#Cpp       7.585    7.954   8.58704   8.159       8.733    16.810    50   b

# Do at least 2 tests for fitLASSOstandardized_seq function below. You are checking output agreements on at least 2 separate inputs
#################################################
# Test Case 1: User-supplied lambda sequence (5 values)
cat("Test 1 - User-supplied lambda sequence (5 values):\n")
lambda_seq1 <- c(1.0, 0.5, 0.3, 0.1, 0.05)

r_result_seq1 <- fitLASSOstandardized_seq(Xtilde, Ytilde, lambda_seq1, eps = 0.001)
cpp_result_seq1 <- fitLASSOstandardized_seq_c(Xtilde, Ytilde, lambda_seq1, eps = 0.001)

cat("R beta_mat dimensions:", dim(r_result_seq1$beta_mat), "\n")
# R beta_mat dimensions: 20 5 
cat("C++ beta_mat dimensions:", dim(cpp_result_seq1), "\n")
# C++ beta_mat dimensions: 20 5  
cat("Beta matrices match:", all.equal(r_result_seq1$beta_mat, cpp_result_seq1), "\n")
#  TRUE  
cat("Max difference:", max(abs(r_result_seq1$beta_mat - cpp_result_seq1)), "\n\n")
# Max difference: 5.724587e-17 
cat("Non-zero coefficients per lambda (R):", colSums(abs(r_result_seq1$beta_mat) > 1e-8), "\n")
# Non-zero coefficients per lambda (R): 0 0 0 3 15 
cat("Non-zero coefficients per lambda (C++):", colSums(abs(cpp_result_seq1) > 1e-8), "\n\n")
# Non-zero coefficients per lambda (C++): 0 0 0 3 15 

# Test Case 2: Longer lambda sequence (10 values)
cat("Test 2 - Longer lambda sequence (30 values):\n")
lambda_seq2 <- exp(seq(log(2.0), log(0.01), length.out = 30))

r_result_seq2 <- fitLASSOstandardized_seq(Xtilde, Ytilde, lambda_seq2, eps = 0.001)
cpp_result_seq2 <- fitLASSOstandardized_seq_c(Xtilde, Ytilde, lambda_seq2, eps = 0.001)

cat("R beta_mat dimensions:", dim(r_result_seq2$beta_mat), "\n")
# R beta_mat dimensions: 20 30 
cat("C++ beta_mat dimensions:", dim(cpp_result_seq2), "\n")
# C++ beta_mat dimensions: 20 30  
cat("Beta matrices match:", all.equal(r_result_seq2$beta_mat, cpp_result_seq2), "\n")
# Beta matrices match: TRUE  
cat("Max difference:", max(abs(r_result_seq2$beta_mat - cpp_result_seq2)), "\n\n")
# Max difference: 1.387779e-16 


# Do microbenchmark on fitLASSOstandardized_seq vs fitLASSOstandardized_seq_c
######################################################################
lambda_seq_bench <- exp(seq(log(1.5), log(0.01), length.out = 30))
mb2 <- microbenchmark(
  R = fitLASSOstandardized_seq(Xtilde, Ytilde, lambda_seq_bench, eps = 0.001),
  Cpp = fitLASSOstandardized_seq_c(Xtilde, Ytilde, lambda_seq_bench, eps = 0.001),
  times = 20
)
print(mb2)
# Unit: microseconds
#expr      min        lq      mean   median       uq      max neval cld
#R 2858.889 2901.7545 3302.0539 2966.329 3143.839 8322.631    20  a 
#Cpp  140.917  141.2655  145.3409  143.254  145.591  166.173    20   b

# Tests on riboflavin data
##########################
require(hdi) # this should install hdi package if you don't have it already; otherwise library(hdi)
data(riboflavin) # this puts list with name riboflavin into the R environment, y - outcome, x - gene erpression

# Make sure riboflavin$x is treated as matrix later in the code for faster computations
class(riboflavin$x) <- class(riboflavin$x)[-match("AsIs", class(riboflavin$x))]

# Standardize the data
out <- standardizeXY(riboflavin$x, riboflavin$y)

# This is just to create lambda_seq, can be done faster, but this is simpler
outl <- fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, n_lambda = 30)

# The code below should assess your speed improvement on riboflavin data
microbenchmark(
  fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, outl$lambda_seq),
  fitLASSOstandardized_seq_c(out$Xtilde, out$Ytilde, outl$lambda_seq),
  times = 10
)

#Unit: milliseconds
#expr       min
#fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, outl$lambda_seq) 975.35540
#fitLASSOstandardized_seq_c(out$Xtilde, out$Ytilde, outl$lambda_seq)  20.68294
#lq       mean    median         uq        max neval cld
#99.31096 1019.21647 1010.9181 1036.40780 1092.98235    10  a 
#20.71447   20.93389   20.8864   21.09909   21.44394    10   b







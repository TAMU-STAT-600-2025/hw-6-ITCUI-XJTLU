# [ToDo] Standardize X and Y: center both X and Y; scale centered X
# X - n x p matrix of covariates
# Y - n x 1 response vector
standardizeXY <- function(X, Y){
  # get the size parameters:n and p
  n <- nrow(X)
  p <- ncol(X)
  
  # [ToDo] Center Y
  Ymean <- mean(Y)
  Ytilde <- Y - Ymean
  
  # [ToDo] Center and scale X
  Xmeans <- colMeans(X)
  X_centered <- sweep(X, 2, Xmeans, FUN = "-") 
  weights <- sqrt(colSums(X_centered^2) / n)

  Xtilde <- sweep(X_centered, 2, weights, FUN = "/") ## Scale X ----
  
  # Return:
  # Xtilde - centered and appropriately scaled X
  # Ytilde - centered Y
  # Ymean - the mean of original Y
  # Xmeans - means of columns of X (vector)
  # weights - defined as sqrt(X_j^{\top}X_j/n) after centering of X but before scaling
  return(list(Xtilde = Xtilde, Ytilde = Ytilde, Ymean = Ymean, Xmeans = Xmeans, weights = weights))
}

# [ToDo] Soft-thresholding of a scalar a at level lambda 
# [OK to have vector version as long as works correctly on scalar; will only test on scalars]
soft <- function(a, lambda){
  return(sign(a) * pmax(abs(a) - lambda, 0))
}

# [ToDo] Calculate objective function of lasso given current values of Xtilde, Ytilde, beta and lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba - tuning parameter
# beta - value of beta at which to evaluate the function
lasso <- function(Xtilde, Ytilde, beta, lambda){
  n <- nrow(Xtilde)
  residual <- Ytilde - Xtilde %*% beta
  rss_term <- sum(residual^2) / (2 * n)
  penalty_term <- lambda * sum(abs(beta))
  
  objective <- rss_term + penalty_term
  return(objective)
}

# [ToDo] Fit LASSO on standardized data for a given lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1 (vector)
# lamdba - tuning parameter
# beta_start - p vector, an optional starting point for coordinate-descent algorithm
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized <- function(Xtilde, Ytilde, lambda, beta_start = NULL, eps = 0.001){
  #[ToDo]  Check that n is the same between Xtilde and Ytilde
  n <- nrow(Xtilde)
  if(n != length(Ytilde)){
    stop("Number of rows of Xtilde is not equal to the length of Ytilde")
  }
  p <- ncol(Xtilde)
  #[ToDo]  Check that lambda is non-negative
  if(lambda <0){
    stop("Lambda must be non-negative")
  }
  
  #[ToDo]  Check for starting point beta_start. 
  # If none supplied, initialize with a vector of zeros.
  # If supplied, check for compatibility with Xtilde in terms of p
  if(is.null(beta_start)){
    beta <- rep(0, p)
  }else{
    # check the compatibility with Xtilde in terms of p
    if(length(beta_start) != p){
      stop("Length of beta_start should be equal with the number of columns of Xtilde")
    }
    beta <- beta_start
  }
  
  #[ToDo]  Coordinate-descent implementation. 
  # Stop when the difference between objective functions is less than eps for the first time.
  # For example, if you have 3 iterations with objectives 3, 1, 0.99999,
  # your should return fmin = 0.99999, and not have another iteration
  aj <- colSums(Xtilde * Xtilde) / n
  r <- as.vector(Ytilde - Xtilde %*% beta)
  f_prev <- lasso(Xtilde, Ytilde, beta, lambda)
  
  max_iter <- 5000L
  for (iter in 1:max_iter) {
    for (j in 1:p) {
      # get r = Y-X_{-j} beta_{-j}
      r <- r + Xtilde[,j] * beta[j]
      
      # update beta_j
      bj <- sum(Xtilde[, j] * r) / n
      beta_new_j <- if (aj[j] > 0) soft(bj, lambda) / aj[j] else 0
      
      # update residual
      r <- r - Xtilde[,j] * beta_new_j
      beta[j] <- beta_new_j
    }
    
    # check objective function
    f_curr <- lasso(Xtilde, Ytilde, beta, lambda)
    if((f_prev - f_curr) < eps){
      fmin <- f_curr
      return(list(beta = beta, fmin = fmin))
    }
    f_prev <- f_curr
  }
  
  fmin <- lasso(Xtilde, Ytilde, beta, lambda)
  # Return 
  # beta - the solution (a vector)
  # fmin - optimal function value (value of objective at beta, scalar)
  return(list(beta = beta, fmin = fmin))
}

# [ToDo] Fit LASSO on standardized data for a sequence of lambda values. Sequential version of a previous function.
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence,
#             is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized_seq <- function(Xtilde, Ytilde, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  # [ToDo] Check that n is the same between Xtilde and Ytilde
  n <- nrow(Xtilde)
  p <- ncol(Xtilde)
  if (n != length(Ytilde)){
    stop("the number of rows of  Xtilde must be the same as the length of Ytilde")
  }
   
  # [ToDo] Check for the user-supplied lambda-seq (see below)
  # If lambda_seq is supplied, only keep values that are >= 0,
  # and make sure the values are sorted from largest to smallest.
  # If none of the supplied values satisfy the requirement,
  # print the warning message and proceed as if the values were not supplied.
  if( !is.null(lambda_seq)){
    lambda_seq <- lambda_seq[lambda_seq > 0]
    lambda_seq <- sort(unique(lambda_seq), decreasing = TRUE)
    if(length(lambda_seq) == 0L){
      warning("No valid non-negative lambdas supplied; will construct default sequence.")
      lambda_seq <- NULL
    }
  }

  # If lambda_seq is not supplied, calculate lambda_max 
  # (the minimal value of lambda that gives zero solution),
  # and create a sequence of length n_lambda as
  # lambda_seq = exp(seq(log(lambda_max), log(0.01), length = n_lambda))
  if (is.null(lambda_seq)){
    XY <- colSums(Xtilde * Ytilde) / n
    lambda_max <- max(abs(XY))
    if(!is.finite(lambda_max) || lambda_max <=0 ){
      lambda_seq <- rep(0, n_lambda)
    }else{
      lambda_seq <- exp(seq(log(lambda_max), log(0.01), length.out = n_lambda))
    }
  }
  
  L <- length(lambda_seq)
  beta_mat <- matrix(0.0, nrow = p, ncol=L)
  fmin_vec <- numeric(L)
  
  beta_start <- rep(0.0, p)
  
  for (k in seq_len(L)){
    lam <- lambda_seq[k]
    fitk <- fitLASSOstandardized(Xtilde, Ytilde, lam, beta_start, eps)
    beta_mat[,k] <- fitk$beta
    fmin_vec[k] <- fitk$fmin
    beta_start <- fitk$beta
  }
  
  # [ToDo] Apply fitLASSOstandardized going from largest to smallest lambda 
  # (make sure supplied eps is carried over). 
  # Use warm starts strategy discussed in class for setting the starting values.
  
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value
  # fmin_vec - length(lambda_seq) vector of corresponding objective function values at solution
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, fmin_vec = fmin_vec))
}

# [ToDo] Fit LASSO on original data using a sequence of lambda values
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  # [ToDo] Center and standardize X,Y based on standardizeXY function
  std <- standardizeXY(X, Y)
  Xtilde <- std$Xtilde
  Ytilde <- std$Ytilde
  Ymean <- std$Ymean
  Xmeans <- std$Xmeans
  weights <- std$weights
 
  # [ToDo] Fit Lasso on a sequence of values using fitLASSOstandardized_seq
  # (make sure the parameters carry over)
  fit_std <- fitLASSOstandardized_seq(Xtilde = Xtilde, Ytilde = Ytilde,
                                      lambda_seq = lambda_seq, n_lambda = n_lambda, eps = eps)
  lambda_seq_used <- fit_std$lambda_seq
  beta_std <- fit_std$beta_mat 
  
  p <- ncol(X)
  L <- length(lambda_seq_used)
  beta_mat <- matrix(0.0, nrow = p, ncol = L)
 
  # [ToDo] Perform back scaling and centering to get original intercept and coefficient vector
  # for each lambda
  
  wpos <- (weights > 0)
  if(any(wpos)){
    beta_mat[wpos,] <- beta_std[wpos,,drop=FALSE] / weights[wpos]
  }
  beta0_vec <- Ymean - crossprod(Xmeans, beta_mat)
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  return(list(lambda_seq = lambda_seq_used, beta_mat = beta_mat, beta0_vec = beta0_vec))
}


# [ToDo] Fit LASSO and perform cross-validation to select the best fit
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# k - number of folds for k-fold cross-validation, default is 5
# fold_ids - (optional) vector of length n specifying the folds assignment (from 1 to max(folds_ids)), if supplied the value of k is ignored 
# eps - precision level for convergence assessment, default 0.001
cvLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, k = 5, fold_ids = NULL, eps = 0.001){
  # [ToDo] Fit Lasso on original data using fitLASSO
  # browser()
  n <- nrow(X)
  p <- ncol(X)
  
  whole_fit <- fitLASSO(X, Y, lambda_seq = lambda_seq, n_lambda = n_lambda, eps = eps)
  lambda_seq_used <- whole_fit$lambda_seq   # length L
  L <- length(lambda_seq_used)
  beta_mat <- whole_fit$beta_mat            # p x L (on original scale)
  beta0_vec <- whole_fit$beta0_vec          # length L
  
  # [ToDo] If fold_ids is NULL, split the data randomly into k folds.
  # If fold_ids is not NULL, split the data according to supplied fold_ids.
  if(is.null(fold_ids)){
    fold_ids <- sample(rep(1:k, length.out = n), size = n)
  }else{
    if (length(fold_ids) != n) stop("fold_ids must have length n.")
    k <- max(fold_ids)
  }
  
  # [ToDo] Calculate LASSO on each fold using fitLASSO,
  # and perform any additional calculations needed for CV(lambda) and SE_CV(lambda)
  cv_errors <- matrix(NA_real_, nrow = L, ncol = k)
  for (fold in 1:k) {
    idx_val <- which(fold_ids == fold)
    idx_tr <- setdiff(seq_len(n), idx_val)
    
    # get train and valid data
    X_tr <- X[idx_tr, ,drop=FALSE]
    Y_tr <- Y[idx_tr]
    X_val <- X[idx_val, ,drop=FALSE]
    Y_val <- Y[idx_val]
    
    # Fit model
    fit_tr <- fitLASSO(X_tr, Y_tr, lambda_seq_used, n_lambda, eps)
    beta_mat_tr <- fit_tr$beta_mat
    beta0_vec_tr <- fit_tr$beta0_vec
    
    # Predict on valid data and get MSE
    for (ell in 1:L){
      yhat <- as.vector(beta0_vec_tr[ell] + X_val %*% beta_mat_tr[,ell])
      cv_errors[ell, fold] <- mean((Y_val - yhat)^2)
    }
  }
  
  # [ToDo] Find lambda_min
  cvm  <- rowMeans(cv_errors)                  # CV(lambda)
  cvsd <- apply(cv_errors, 1, sd)              # SD across folds
  cvse <- cvsd / sqrt(k)                       # SE_CV(lambda)
  idx_min <- which.min(cvm)
  lambda_min <- lambda_seq_used[idx_min]
  
  # [ToDo] Find lambda_1SE
  thresh <- cvm[idx_min] + cvse[idx_min]
  candidates <- which(cvm <= thresh)
  lambda_1se <- lambda_seq_used[min(candidates)]
  
  # Return output
  # Output from fitLASSO on the whole data
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  # fold_ids - used splitting into folds from 1 to k (either as supplied or as generated in the beginning)
  # lambda_min - selected lambda based on minimal rule
  # lambda_1se - selected lambda based on 1SE rule
  # cvm - values of CV(lambda) for each lambda
  # cvse - values of SE_CV(lambda) for each lambda
  return(list(lambda_seq = lambda_seq_used, beta_mat = beta_mat, beta0_vec = beta0_vec, fold_ids = fold_ids, lambda_min = lambda_min, lambda_1se = lambda_1se, cvm = cvm, cvse = cvse))
}



















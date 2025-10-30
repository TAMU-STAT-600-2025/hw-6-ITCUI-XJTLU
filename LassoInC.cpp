#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// Soft-thresholding function, returns scalar
// [[Rcpp::export]]
double soft_c(double a, double lambda){
  // Your function code goes here
  if (a >  lambda) return a - lambda;
  if (a < -lambda) return a + lambda;
  return 0.0;
}

// Lasso objective function, returns scalar
// [[Rcpp::export]]
double lasso_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& beta, double lambda){
  // Your function code goes here
  
  // numbere of sample 
  int n = Xtilde.n_rows;
  
  // residual
  arma::colvec residual = Ytilde - Xtilde * beta;
  
  //RSS term 
  double rss_term = arma::accu(arma::square(residual)) / (2.0*n); 
  
  //penalty 
  double penalty_term = lambda * arma::accu(arma::abs(beta));
  
  double objective = rss_term + penalty_term;
  
  return objective;
}

// Lasso coordinate-descent on standardized data with one lamdba. Returns a vector beta.
// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, double lambda, const arma::colvec& beta_start, double eps = 0.001){
  // Your function code goes here
  int n = Xtilde.n_rows;  
  int p = Xtilde.n_cols;
  arma::colvec beta = beta_start;
  
  // Precompute aj = sum(Xtilde[,j]^2) / n for each column
  arma::vec aj(p);
  for (int j=0; j < p; j++){
    aj(j) = arma::accu(arma::square(Xtilde.col(j))) / n;
  }
  
  // initialize residual 
  arma::colvec r = Ytilde - Xtilde * beta;
  
  // initial beta function 
  double f_prev = lasso_c(Xtilde, Ytilde, beta, lambda);
  
  int max_iter = 5000;
  // coordinate descent loop 
  for (int iter =0; iter < max_iter; iter++){
    for(int j=0; j < p; j++){
      // add back the contribution of beta[j]
      r += Xtilde.col(j) * beta(j);
  
      // compute bj = <Xtilde[,j], r> / n
      double bj = arma::dot(Xtilde.col(j), r) / n;
      
      // compute beta[j] useing soft-threshold
      double beta_new_j = 0.0;
      if(aj(j) >0){
        beta_new_j = soft_c(bj, lambda) / aj(j);
      }
      
      // Update residule by removing new contribution 
      r -= Xtilde.col(j) * beta_new_j;
      
      // update beta[j]
      beta(j) = beta_new_j;
    }
    
    // check convergence
    double f_curr = lasso_c(Xtilde, Ytilde, beta, lambda);
    if (f_prev - f_curr < eps){
      break;
    }
    f_prev = f_curr;
  }
  return beta;
}  

// Lasso coordinate-descent on standardized data with supplied lambda_seq. 
// You can assume that the supplied lambda_seq is already sorted from largest to smallest, and has no negative values.
// Returns a matrix beta (p by number of lambdas in the sequence)
// [[Rcpp::export]]
arma::mat fitLASSOstandardized_seq_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& lambda_seq, double eps = 0.001){
  // Your function code goes here
  int n = Xtilde.n_rows;
  int p = Xtilde.n_cols;
  int nlambda = lambda_seq.n_elem;
  
  // Initialize output matrix: p x nlambda
  arma::mat beta_matrix(p, nlambda, arma::fill::zeros);
  
  // Precompute aj = sum(Xtilde[,j]^2) / n for each column
  arma::vec aj(p);
  for (int j = 0; j < p; j++) {
    aj(j) = arma::accu(arma::square(Xtilde.col(j))) / n;
  }
  
  // Initialize beta with zeros (warm start for first lambda)
  arma::colvec beta(p, arma::fill::zeros);
  
  int max_iter = 5000;
  
  // Loop over lambda sequence (from largest to smallest)
  for (int l = 0; l < nlambda; l++) {
    double lambda = lambda_seq(l);
    
    // Initialize residual: r = Y - X * beta
    arma::colvec r = Ytilde - Xtilde * beta;
    
    // Compute initial objective function
    double f_prev = lasso_c(Xtilde, Ytilde, beta, lambda);
    
    // Coordinate descent loop
    for (int iter = 0; iter < max_iter; iter++) {
      for (int j = 0; j < p; j++) {
        // Add back the contribution of beta[j]
        r += Xtilde.col(j) * beta(j);
        
        // Compute bj = <Xtilde[,j], r> / n
        double bj = arma::dot(Xtilde.col(j), r) / n;
        
        // Update beta[j] using soft-thresholding
        double beta_new_j = 0.0;
        if (aj(j) > 0) {
          beta_new_j = soft_c(bj, lambda) / aj(j);
        }
        
        // Update residual by removing new contribution
        r -= Xtilde.col(j) * beta_new_j;
        
        // Update beta[j]
        beta(j) = beta_new_j;
      }
      
      // Check convergence
      double f_curr = lasso_c(Xtilde, Ytilde, beta, lambda);
      if (f_prev - f_curr < eps) {
        break;
      }
      f_prev = f_curr;
    }
    
    // Store solution for this lambda
    beta_matrix.col(l) = beta;
    // Note: beta is carried over as starting point for next lambda (warm start)
  }
  
  return beta_matrix;
}






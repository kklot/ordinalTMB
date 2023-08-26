#include <TMB.hpp>

// Only for logit link

template <class Type>
Type CDF_polr(Type x){return invlogit(x);}
/** 
* Pointwise (pw) log-likelihood vector
*
* @param y The integer outcome variable.
* @param eta A vector of linear predictors
* @param cutpoints An ordered vector of cutpoints
* @param link An integer indicating the link function
* @return A vector of log-likelihods
*/
template <class Type>
vector<Type> pw_polr(vector<Type> y, vector<Type> eta, vector<Type> cutpoints) {
  int N = eta.size(), J = cutpoints.size() + 1;
  vector<Type> ll(N);
  for (int n=0; n < N; n++) {
    if (y[n] == 1) 
      ll[n] = CDF_polr(cutpoints[1] - eta[n]);
    elseif (y[n] == J) 
      ll[n] = 1 - CDF_polr(cutpoints[J - 1] - eta[n]);
    else 
      ll[n] = CDF_polr(cutpoints[y[n]] - eta[n]) - CDF_polr(cutpoints[y[n] - 1] - eta[n]);
  }
  return log(ll);
}
/**
* Map from conditional probabilities to cutpoints
*
* @param probabilities A J-simplex
* @param scale A positive scalar
* @return A vector of length J - 1 whose elements are in increasing order
*/
template <class Type>
vector<Type> make_cutpoints(vector<Type> probabilities, Type scale = 1.0) {
  int C = probabilities.size() - 1; 
  vector<Type> cutpoints(C);
  Type running_sum = 0.0;
  for (int c=0; c < C; c++) {
    running_sum += probabilities[c];
    cutpoints[c] = logit(running_sum);
  }
  return scale * cutpoints;
}

// transform unbounded to a simplex 
template<class Type>
vector<Type> simplex_transform(vector<Type> y) {
  int K = y.size() + 1;
  vector<Type> z(K-1); z.setZero();
  for (int k = 0; k < K - 1; k++)
    z[k] = invlogit(y[k] + log(1/(K - (k + 1))) );
  vector<Type> x(K); x.setZero();
  x[0] = z[0];
  for (int k = 1; k < K-1; k++) {
    Type rollsum = 0;
    for (int i = 0; i < k; ++i) rollsum += x[i];
    x[k] = (1 - rollsum) * z[k];
  }
  x[K-1] = 1 - sum(x);
  return x;
}

// transform unbounded to a unit vector: ND if all(y)==0
template<class Type>
vector<Type> as_unit_vector(vector<Type> y) {
  Type norm = pow(y.transpose() * y, 0.5);
  y /= norm;
  return y;
}
?lgamma
// transform unbounded to bounded a-b
template<class Type>
Type as_bounded(Type y, Type l, Type u) { return invlogit(y) * (u - l) + l; }

template<class Type>
Type dirichlet_lpdf(vector<Type> x, vector<Type> alpha) {
  Type t1 = lgamma(alpha.sum()), t2 = 0, t3 = 0;
  for (int i = 0; i < alpha.size(); i++) {
    t2 += lgamma(alpha[i]);
    t3 += (alpha[i] - 1) * log(x[i]);
  }
  return t1 - t2 + t3;
}

template <class Type>
Type objective_function<Type>::operator()() {
  Type target = 0.0;

  // Meta data
  DATA_SCALAR(K);
  Type half_K = K * 0.5;
  DATA_SCALAR(J); // 5

  DATA_VECTOR(y);
  DATA_MATRIX(X);
  Type N = y.size();
  
  PARAMETER_VECTOR(pi_norm); // length J-1, then transform to simplex J
  vector<Type> pi = simplex_transform(pi_norm); // length J

  // https://mc-stan.org/docs/reference-manual/unit-vector.html
  PARAMETER_VECTOR(u_norm); // length K then transform to unit vector
  vector<Type> u = as_unit_vector(u_norm);
  
  PARAMETER(R2_norm); // https://mc-stan.org/docs/reference-manual/logit-transform-jacobian.html
  Type R2_low = (K > 1) ? 0 : -1;
  Type R2 = as_bounded(R2_norm, R2_low, Type(1));
  Type Delta_y = 1 / pow(1 - R2, 0.5);
  
  // transform parameters
  vector<Type> beta(K);
  beta = u[0] * pow(R2, 0.5) * Delta_y * pow(N - 1, 0.5);

  vector<Type> cutpoints(J - 1);
  cutpoints = make_cutpoints(pi, Delta_y);

  // hyperparameter
  real<lower=0> regularization;
  DATA_VECTOR(prior_counts); // J drawn from dirichlet
  // vector<lower=0>[J] prior_counts;

  // linear predictor
  vector<Type> eta = X * beta;
  vector<Type> pwll = pw_polr(y, eta, cutpoints);

  target -= pwll.sum();
  target -= dirichlet_lpdf(pi, prior_counts);
  REPORT(cutpoints);
  REPORT(beta);  
  REPORT(pwll);

  return target;
}

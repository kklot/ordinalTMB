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
  int N = y.size(), J = cutpoints.size() + 1;
  vector<Type> ll(N);
  for (int n=0; n < N; n++) {
    int j = asDouble(y[n]); //raw data
    if (j == 1)
      ll[n] = CDF_polr(cutpoints[1 - 1] - eta[n]);
    else if (j == J) 
      ll[n] = 1 - CDF_polr(cutpoints[J - 1 - 1] - eta[n]);
    else 
      ll[n] = CDF_polr(cutpoints[j - 1] - eta[n]) - CDF_polr(cutpoints[j - 1 - 1] - eta[n]);
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
    z[k] = invlogit(y[k] + log(1.0 / (K - (k + 1.0))) );
  vector<Type> x(K); x.setZero();
  x[0] = z[0];
  for (int k = 1; k < K - 1; k++)
    x[k] = (1 - x.head(k).sum()) * z[k]; // head index is different
  x[K-1] = 1 - sum(x);
  return x;
}

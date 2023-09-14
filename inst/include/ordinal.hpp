// Copyright (c) 2023 Kinh Nguyen

// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.


/**
 * Prior for correlation
 * 
 * See here:
 * https://inla.r-inla-download.org/r-inla.org/doc/prior/betacorrelation.pdf
*/
template <class Type>
Type beta_correlation_lpdf(Type theta, Type a = 2.0, Type b = 2.0) {
    Type 
      e_theta = exp(theta), 
      p = e_theta / (1 + e_theta),
      ld = dbeta(p, a, b, true) + theta - 2.0 * log(1.0 + e_theta);
    return ld;
}
// Transform to correlation
template <class Type>
Type theta_to_rho(Type theta) {
  Type rho = (exp(theta) - 1) / (exp(theta) + 1);
  return rho;
}
/**
 * Fill covariance matrix with a vector of correlation 
*/
template <class Type>
matrix<Type> fill_covariance(matrix<Type> M, vector<Type> v) {
  int n_col = M.cols(), count = 0; // TODO check square, check v's len
  for (int r = 0; r < n_col - 1; ++r)
    for (int c = 0; c < n_col; ++c) {
      M(r,c) = v[count];
      M(c,r) = v[count];
      ++count;
    }
  return M;
}
/**
 * Cummulative link for ordinal
 * 
 * Only logit and probit for now, to do multivariate model
 * 
*/
template <class Type>
Type CDF_polr(Type x, int link = 1) {
  // link 1 logit 2 probit
  Type cum = 0;
  if (link == 1) cum = invlogit(x);
  if (link == 2) cum = pnorm(x);
  return cum;
}
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
vector<Type> pw_polr(vector<Type> y, vector<Type> eta, vector<Type> cutpoints, int link = 1) {
  int N = y.size(), J = cutpoints.size() + 1;
  vector<Type> ll(N);
  for (int n=0; n < N; n++) {
    int j = asDouble(y[n]); //raw data
    if (j == 1)
      ll[n] = CDF_polr(cutpoints[1 - 1] - eta[n], link);
    else if (j == J) 
      ll[n] = 1 - CDF_polr(cutpoints[J - 1 - 1] - eta[n], link);
    else 
      ll[n] = CDF_polr(cutpoints[j - 1] - eta[n], link) - CDF_polr(cutpoints[j - 1 - 1] - eta[n], link);
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
vector<Type> make_cutpoints(vector<Type> probabilities, int link = 1) {
  int C = probabilities.size() - 1; 
  vector<Type> cutpoints(C);
  Type running_sum = 0.0;
  if (link == 1)
    for (int c=0; c < C; c++) {
      running_sum += probabilities[c];
      cutpoints[c] = logit(running_sum);
    } 
  if (link == 2) 
    for (int c=0; c < C; c++) {
      running_sum += probabilities[c];
      cutpoints[c] = qnorm(running_sum);
    }
  return cutpoints;
}

/** 
 * transform unbounded to a simplex for paraneter in ordinal model
*/
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

/**
 * Export Fortran function and register for TMB
 * 
 * See Kasper's email
*/
extern "C" {
  double F77_NAME(mvbvn)(void *lower, void *upper, void *infin, void *correl);
  const static R_CallMethodDef R_CallDef[] = {
    TMB_CALLDEFS,
    {"mvbvn", (DL_FUNC) &F77_NAME(mvbvn), 4},
    {NULL, NULL, 0}
  };
  void R_init_ordinalTMB(DllInfo *dll) 
  {
    R_registerRoutines(dll, NULL, R_CallDef, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    TMB_CCALLABLES("ordinalTMB");
  }
}

namespace katomic 
{
  using atomic::tiny_ad::isfinite;
  template<class Float>
  struct BVN_t
  {
    typedef Float Scalar;
    Float y1, y2; // data
    Float c1, c2, c3, c4, c5, c6, c7, c8; // eight cutpoints
    Float eta, rho;
    // Evaluate the cumulative bivariate density 
    Float operator() () {
      Float ans = 0;
      
      int N_RESPONSES = 5, N_CUTS = N_RESPONSES - 1;

      // make cutpoints
      vector<Float> c_pi0(N_CUTS);
      vector<Float> c_pi1(N_CUTS);
      c_pi0 << c1, c2, c3, c4;
      c_pi1 << c5, c6, c7, c8;
      vector<Float> pi0 = simplex_transform(c_pi0);
      vector<Float> pi1 = simplex_transform(c_pi1);
      // find cuts
      vector<Float> C0(6), C1(6);
      C0[0] = -FLT_MIN; C0[5] = FLT_MAX;
      C1[0] = -FLT_MIN; C1[5] = FLT_MAX;
      CppAD::vector<double> cum_sum0(1), cum_sum1(1);
      cum_sum0[0] = 0.;
      cum_sum1[0] = 0.;
      for (int c = 1; c <= 4; c++) {
        cum_sum0[0] += asDouble(pi0[c - 1]);
        cum_sum1[0] += asDouble(pi1[c - 1]);
        C0[c] = qnorm(cum_sum0[0]);
        C1[c] = qnorm(cum_sum1[0]);
      }

      int y1i = asDouble(y1), y2i = asDouble(y2);
      Float 
        ly1 = C0[y1i - 1] - eta,
        uy1 = C0[y1i    ] - eta,
        ly2 = C1[y2i - 1] - eta,
        uy2 = C1[y2i    ] - eta, 
        lower_null[2] = {FLT_MIN, FLT_MIN}, 
        infin_null[2] = {0, 0},
        upper[2] = {uy1, uy2};

      // Rectangle probs
      Float p1 = mvbvn_(lower_null, upper, infin_null, &rho);
      upper[0] = ly1;
      upper[1] = uy2;
      Float p2 = mvbvn_(lower_null, upper, infin_null, &rho);
      upper[0] = uy1;
      upper[1] = ly2;
      Float p3 = mvbvn_(lower_null, upper, infin_null, &rho);
      upper[0] = ly1;
      upper[1] = ly2;
      Float p4 = mvbvn_(lower_null, upper, infin_null, &rho);
      p1 = (!isfinite(p1)) ? 0 : p1;
      p2 = (!isfinite(p2)) ? 0 : p2;
      p3 = (!isfinite(p3)) ? 0 : p3;
      p4 = (!isfinite(p4)) ? 0 : p4;
      Float pr = p1 - p2 - p3 + p4;
      pr = (pr < 0) ? 0 : pr; // what to do better?
      ans += pr;
      return ans;
    };
  };
  
  template<class Float>
  Float eval(
    Float y1, Float y2,
    Float c1, Float c2, Float c3, Float c4, Float c5, Float c6, Float c7, Float c8,
    Float eta, Float rho) 
  {
    BVN_t<Float> f = {y1, y2, c1, c2, c3, c4, c5, c6, c7, c8, eta, rho};
    return f();
  }
  
  TMB_BIND_ATOMIC
  (
    func, 
    000000000011, // need at least one to compile
    eval(
      x[0], x[1], 
      x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], 
      x[10], x[11]
    )
  )
  
  template<class Type>
  Type BVN(
    Type y1, Type y2,
    Type c1, Type c2, Type c3, Type c4, Type c5, Type c6, Type c7, Type c8,
    Type eta, Type rho) 
  {
    vector<Type> args(13); // Last index reserved for derivative order
    args << y1, y2, c1, c2, c3, c4, c5, c6, c7, c8, eta, rho, 0; 
    return katomic::func(CppAD::vector<Type>(args))[0];
  }
}

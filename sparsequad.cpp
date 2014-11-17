#include "mex.h"
#include "blas.h"
#include "matrix.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <emmintrin.h>

/*
 * compilation, at matlab prompt: (adjust NUM_THREADS as appropriate)
 * 
 * == windows ==
 * 
 * mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=4 sparsequad.cpp
 * 
 * == linux ==
 * 
 * mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=4 -v CXXFLAGS='$CXXFLAGS -std=c++0x -fPIC' sparsequad.cpp
 */


#define LTIC_MATRIX_PARAMETER_IN        prhs[0]
#define DIAG_W_VECTOR_PARAMETER_IN      prhs[1]
#define RTIC_MATRIX_PARAMETER_IN        prhs[2]
#define ZTIC_MATRIX_PARAMETER_IN        prhs[3]

// Y = Z'*R'*diag(W)*L

static void
sparsequad (const mxArray* prhs[],
            double*        Y,
            double*        ZticR,
            size_t         start,
            size_t         end)
{
  mwIndex* Lticir = mxGetIr(LTIC_MATRIX_PARAMETER_IN);
  mwIndex* Lticjc = mxGetJc(LTIC_MATRIX_PARAMETER_IN);
  double* Ltics = mxGetPr(LTIC_MATRIX_PARAMETER_IN);

  double* W = mxGetPr(DIAG_W_VECTOR_PARAMETER_IN);

  mwIndex* Rticir = mxGetIr(RTIC_MATRIX_PARAMETER_IN);
  mwIndex* Rticjc = mxGetJc(RTIC_MATRIX_PARAMETER_IN);
  double* Rtics = mxGetPr(RTIC_MATRIX_PARAMETER_IN);

  size_t k = mxGetM(ZTIC_MATRIX_PARAMETER_IN);
  double* Ztic = mxGetPr(ZTIC_MATRIX_PARAMETER_IN);

  for (size_t n = start; n < end; ++n) {
    double wi = W[n];

    memset (ZticR, 0, k * sizeof (double));

    mwIndex Rticstop = Rticjc[n + 1];
    for (mwIndex j = Rticjc[n]; j < Rticstop; ++j) {
      double *Zticdr = Ztic + Rticir[j] * k;
      double Rs = wi * Rtics[j];

      for (size_t i = 0; i < k; ++i) {
        ZticR[i] += Rs * Zticdr[i];
      }
    }

    mwIndex Lticstop = Lticjc[n + 1];

    for (mwIndex j = Lticjc[n]; j < Lticstop; ++j) {
      double *Yout = Y + Lticir[j] * k;
      double Ls = Ltics[j];

      for (size_t i = 0; i < k; ++i) {
        Yout[i] += Ls * ZticR[i];
      }
    }
  }
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  if (first) {
    mexPrintf("sparsequad using NUM_THREADS=%u\n",NUM_THREADS);
    first=0;
  }

  switch (nrhs) {
    case 4:
      if (! mxIsSparse (LTIC_MATRIX_PARAMETER_IN) || 
          mxIsSparse (DIAG_W_VECTOR_PARAMETER_IN) || 
          ! mxIsSparse (RTIC_MATRIX_PARAMETER_IN) || 
          mxIsSparse (ZTIC_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Parameters must be sparse, dense, sparse, and dense.");
        return;
      }

      if (mxGetN(ZTIC_MATRIX_PARAMETER_IN) != mxGetM(RTIC_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Ztic and Rtic have incompatible shape.");
        return;
      }

      if (mxGetN(RTIC_MATRIX_PARAMETER_IN) != mxGetN(DIAG_W_VECTOR_PARAMETER_IN)) {
        mexErrMsgTxt("Rtic and W have incompatible shape.");
        return;
      }

      if (mxGetN(DIAG_W_VECTOR_PARAMETER_IN) != mxGetN(LTIC_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("W and Ltic have incompatible shape.");
        return;
      }

      break;

    default:
      mexErrMsgTxt("Wrong number of arguments.");
      return;
  }

  size_t k = mxGetM(ZTIC_MATRIX_PARAMETER_IN);
  size_t dl = mxGetM(LTIC_MATRIX_PARAMETER_IN);
  size_t n = mxGetN(LTIC_MATRIX_PARAMETER_IN);

  plhs[0] = mxCreateDoubleMatrix(k, dl, mxREAL);
  double* Y = mxGetPr(plhs[0]);

  std::thread t[NUM_THREADS];
  double* s[NUM_THREADS];
  size_t quot = n/NUM_THREADS;

  for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
    s[i] = (double*) mxCalloc((dl + 1) * k, sizeof(double));
  }

  double* ZticR = (double*) mxCalloc (k, sizeof (double));

  for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i] = std::thread(sparsequad,
                       prhs,
                       s[i] + k,
                       s[i],
                       i * quot,
                       (i + 1) * quot);

  }

  sparsequad (prhs, Y, ZticR, (NUM_THREADS - 1) * quot, n);

  mxFree (ZticR);

  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    double oned = 1.0;
    ptrdiff_t one = 1;
    ptrdiff_t dltimesk = dl * k;

    t[i].join ();
    daxpy (&dltimesk, &oned, s[i] + k, &one, Y, &one);
    mxFree(s[i]);
  }

  return;
}

#include "mex.h"
#include "matrix.h"
#include <cmath>
#include <cstdint>
#include <thread>
#include <emmintrin.h>

/*
 * compilation, at matlab prompt: (adjust NUM_THREADS as appropriate)
 * 
 * == windows ==
 * 
 * mex OPTIMFLAGS="/Ox" -largeArrayDims -DNUM_THREADS=4 dmsm.cpp
 * 
 * == linux ==
 * 
 * mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=4 -v CXXFLAGS='$CXXFLAGS -std=c++0x -fPIC' dmsm.cpp
 */


#define DENSE_MATRIX_PARAMETER_IN     prhs[0]
#define SPARSE_MATRIX_PARAMETER_IN    prhs[1]

// X = D*S => X' = S'*D'

static void
sparsetic_times_densetic (const mxArray* prhs[], mxArray* plhs[], size_t start, size_t end)
{
  mwIndex* ir = mxGetIr(SPARSE_MATRIX_PARAMETER_IN);       /* Row indexing      */
  mwIndex* jc = mxGetJc(SPARSE_MATRIX_PARAMETER_IN);       /* Column count      */
  double* s  = mxGetPr(SPARSE_MATRIX_PARAMETER_IN);        /* Non-zero elements */
  double* Btic = mxGetPr(DENSE_MATRIX_PARAMETER_IN);
  mwSize Bcol = mxGetM(DENSE_MATRIX_PARAMETER_IN);
  double* Xtic = mxGetPr(plhs[0]);
  mwSize Xcol = mxGetM(plhs[0]);        

  for (size_t i=start; i<end; ++i) {            /* Loop through rows of A (and X) */
    mwIndex stop = jc[i+1];
    for (mwIndex k=jc[i]; k<stop; ++k) {        /* Loop through non-zeros in ith row of A */
      double sk = s[k];
      double* Bticrow = Btic + ir[k] * Bcol;
      double* Xticrow = Xtic + i * Xcol;
      for (mwSize j=0; j<Xcol; ++j) {
        *Xticrow++ += sk * *Bticrow++;
      }
    }
  }
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  if (first) {
    mexPrintf("dmsm using NUM_THREADS=%u\n",NUM_THREADS);
    first=0;
  }

  switch (nrhs) {
    case 2:
      if (! mxIsSparse(SPARSE_MATRIX_PARAMETER_IN) || mxIsSparse(DENSE_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Require one sparse and one dense argument. Fail.");
        return;
      }
      if (mxGetM(SPARSE_MATRIX_PARAMETER_IN) != mxGetN(DENSE_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Arguments have incompatible shape. Fail.");
        return;
      }
      break;

    default:
      mexErrMsgTxt("Wrong number of arguments. Fail.");
      return;
  }

  double* Btic = mxGetPr(DENSE_MATRIX_PARAMETER_IN);
  size_t Bcol = mxGetM(DENSE_MATRIX_PARAMETER_IN);

  size_t Arow = mxGetN(SPARSE_MATRIX_PARAMETER_IN);

  plhs[0] = mxCreateDoubleMatrix(Bcol, Arow, mxREAL);

  double* Xtic = mxGetPr(plhs[0]);

  std::thread t[NUM_THREADS];
  size_t quot = Arow/NUM_THREADS;

  for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i] = std::thread(sparsetic_times_densetic,
                       prhs,
                       plhs,
                       i * quot,
                       (i + 1) * quot);

  }

  sparsetic_times_densetic (prhs, plhs, (NUM_THREADS - 1) * quot, Arow);

  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i].join ();
  }

  return;
}

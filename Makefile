NUM_THREADS=8

ifeq (,$(wildcard /proc/cpuinfo))
all: dmsm.mexw64 sparsequad.mexw64
else
all: dmsm.mexa64 sparsequad.mexa64
endif

clean:
	rm -f $(wildcard *.mexa64) $(wildcard *.mexw64)

%.mexa64: %.cpp
	 matlab -nodisplay -nojvm -r 'mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=$(NUM_THREADS) -v CXXFLAGS='"'"'$$CXXFLAGS -std=c++0x -fPIC'"'"' $*.cpp -lmwblas; exit'

%.mexw64: %.cpp
	 matlab -nodisplay -nojvm -r 'mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=$(NUM_THREADS) $*.cpp -lmwblas; exit'

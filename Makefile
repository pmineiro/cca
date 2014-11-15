NUM_THREADS=8

ifeq (,$(wildcard /proc/cpuinfo))
all: $(patsubst %.cpp,%.mexw64,$(wildcard *.cpp))
else
all: $(patsubst %.cpp,%.mexa64,$(wildcard *.cpp))
endif

clean:
	rm -f $(wildcard *.mexa64) $(wildcard *.mexw64)

%.mexa64: %.cpp
	 matlab -nodisplay -nojvm -r 'mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=$(NUM_THREADS) -v CXXFLAGS='"'"'$$CXXFLAGS -std=c++0x -fPIC'"'"' $*.cpp -lmwblas; exit'

%.mexw64: %.cpp
	 matlab -nodisplay -nojvm -r 'mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=$(NUM_THREADS) $*.cpp -lmwblas; exit'

NUM_THREADS=8

ifeq (,$(wildcard /proc/cpuinfo))
all: dmsm.mexw64
else
all: dmsm.mexa64
endif

clean:
	rm -f $(wildcard dmsm.mexa64) $(wildcard dmsm.mexw64)

dmsm.mexa64: dmsm.cpp
	 matlab -nodisplay -nojvm -r 'mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=$(NUM_THREADS) -v CXXFLAGS='"'"'$$CXXFLAGS -std=c++0x -fPIC'"'"' dmsm.cpp; exit'

dmsm.mexw64: dmsm.cpp
	 matlab -nodisplay -nojvm -r 'mex OPTIMFLAGS="/Ox" -largeArrayDims -DNUM_THREADS=$(NUM_THREADS) -v dmsm.cpp; exit'

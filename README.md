Code Optimization Techniques
==========================

Implementation and benchmark of optimization techniques and algorithms applied to the Matrix Multiplication problem on a CPU/GPU multithreaded environment.

This was an undergraduate research project done at Federal University of Rio de Janeiro (UFRJ) July/2013 - July/2014.

Optimization Techniques
==========================

- Loop unrolling
- Loop blocking
- Z-order memory layout
- SIMD (Single Instruction, Multiple Data) instructions

Multithreading
==========================

- OpenMP   (CPU)
- Pthreads (CPU)
- OpenCL   (GPU)


Algorithms
==========================

- Standard
- Strassen
- Recursive

Library implementations
==========================

- OpenBLAS    (CPU)
- AMD APP SDK (GPU)
- clMath      (GPU)

Test and Benchmark program
==========================

The program <b>test-mult</b> is responsible for setting up test cases, execute different implementations, retrieve timing results and generate final report.
You can call './test-mult M K N' to start the program. Those parameters are the matrices dimensions [M,K] and [K,N], among other parameters:

- Elements data type (int, float, double)
- Blocking size
- Compiler optimization level (-O0, -O1, -O2, -O3, -Ofast -march=native -mfpmath=sse -frename-registers)
- Execution repeating count
- Random number generator range
- Algorithms selection

All selected algorithms/implementations will be compiled using the choosen compiler optimization level. Then they will be executed by a execv() call and the test-mult execution will resume only after its child process is finished.
Timings are collected for reading, processing and writing phases. gettimeofday() is used for multithreaded and library implementations. getrusage() is used on some single thread programs for a more precise timing. Both timing options are reliable.
Not all algorithms will run under some selected parameters (eg. int data type and clMath), so they won't appear on algorithms selection.

Results
=========================
After gathering and analyzing multiple tests results, I could come to some conclusions.
- 1. All algorithms benefit from compiler optimization. There is a considerable performance gain when comparing -O2 against -O0 processing timings. Minor differences from -O2 to -O3 or -Ofast.
- 2. Small gain when changing cache blocking size from 32 to 64. 64 to 128 gives slightly improvements, no effects or worsen performance depending on the algorithm.
- 3. The best algorithm to use to multiply matrices up to dimensions [2048,2048] is the OpenBLAS implementation using multiple threads. Bigger matrices will perform better on GPU environment, where the clMath program gets the best processing time.

Implementations description
=========================

19 programs.

standardMult:
Basic 3 nested loops.

optimizedMult:
Cache blocking (parameter) and loop unrolling x5

optimizedMult2:
Cache blocking (parameter) and SIMD instruction (x8 float, x4 double)

"optimizedMultParallel", 
"optimizedMultParallel2", 
"strassenMult", 
"strassenMult2", 
"strassenMultParallel", 
"strassenMultParallel2", 
"recursiveMult", 
"recursiveMult2", 
"recursiveMultParallel", 
"recursiveMultParallel2", 
"openBlas", 
"openBlasParallel", 
"standardOpenCL", 
"optimizedOpenCL", 
"AMDopenCL", 
"clMath"



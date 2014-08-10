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

This chart represents the total timing for each algorithm execution. OpenBLAS got the fastest runtime, but in fact it is slower than clMath in processing time. The last 3 GPU implementations got a slower writing time, the reason is still unknown and weird for me because in the other tests there isn't this slowdown on writing phase for GPU. You can check all the benchmark files on 'bench' folder.
<img src="https://docs.google.com/spreadsheet/oimg?key=0AjKHxPB2qgJXdFdlVm41M2laSFZSeHRRWGV5bEwwSEE&oid=7&zx=q84mxns399q3" />

Implementations description
=========================

19 programs.

<b>standardMult</b>:<br>
Basic 3 nested loops.

<b>optimizedMult</b>:<br>
Cache blocking (parameter) and loop unrolling x5.

<b>optimizedMult2</b>:<br>
Cache blocking (parameter) and SIMD instruction (x8 float, x4 double).

<b>optimizedMultParallel</b>:<br>
Cache blocking (parameter), loop unrolling x5 and OpenMP 'parallel for' on outermost loop.

<b>optimizedMultParallel2</b>:<br>
Cache blocking (parameter),SIMD instruction (x8 float, x4 double) and OpenMP 'parallel for' on outermost loop.

<b>strassenMult</b>:<br>
Cache blocking (parameter) and loop unrolling x5. Only works on square matrices with dimension as a power of two. Memory saving implementation by using final matrix as temporary matrix.

<b>strassenMult2</b>:<br>
Cache blocking (parameter) and SIMD instruction (x8 float, x4 double). Only works on square matrices with dimension as a power of two. Memory saving implementation by using final matrix as temporary matrix.

<b>strassenMultParallel</b>:<br> 
Cache blocking (parameter), loop unrolling x5 and Pthreads parallel execution on each matrix quadrant. Only works on square matrices with dimension as a power of two. Memory saving implementation by using final matrix as temporary matrix.

<b>strassenMultParallel2</b>:<br> 
Cache blocking (parameter), SIMD instruction (x8 float, x4 double) and Pthreads parallel execution on each matrix quadrant. Only works on square matrices with dimension as a power of two. Memory saving implementation by using final matrix as temporary matrix.

<b>recursiveMult</b>:<br> 
Cache blocking (parameter) and loop unrolling x5. Z-order memory layout for better cache usage.

<b>recursiveMult2</b>:<br> 
Cache blocking (parameter) and SIMD instruction (x8 float, x4 double). Z-order memory layout for better cache usage.

<b>recursiveMultParallel</b>:<br> 
Cache blocking (parameter), loop unrolling x5 and Pthreads parallel execution for generating tasks and executing them. Task Scheduler and Z-order memory layout for better cache usage.

<b>recursiveMultParallel2</b>:<br> 
Cache blocking (parameter), SIMD instruction (x8 float, x4 double) and Pthreads parallel execution for generating tasks and executing them. Task Scheduler and Z-order memory layout for better cache usage.

<b>openBlas</b>:<br> 
OpenBLAS implementation using one thread.<br>
More Info: http://www.openblas.net/

<b>openBlasParallel</b>:<br> 
OpenBLAS implementation using multiple threads.<br>
More Info: http://www.openblas.net/

<b>standardOpenCL</b>:<br> 
Standard OpenCL implementation. Each thread will compute the result for one single matrix cell.

<b>optimizedOpenCL</b>:<br>
OpenCL implementation, blocking and use of group local memory.

<b>AMDopenCL</b>:<br>
Implementation that came with the AMD APP SDK v2.8.1
<br>More info: http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/

<b>clMath</b>:<br>
AMD Accelerated Parallel Processing Math Library.
<br>More info: http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-math-libraries/



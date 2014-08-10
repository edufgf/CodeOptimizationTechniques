#define BLOCK_SIZE 8

__kernel void mat_mult( __global const int *A, __global const int *B, __global int *C, 
						const int M, const int N, const int K) {
    // Block index
    int block_i = get_group_id(0);
    int block_j = get_group_id(1);
 
    // Thread index
    int thread_i = get_local_id(0);
    int thread_j = get_local_id(1);
    
    int acc = 0;
 
    // Index of the first sub-matrix of A processed 
    // by the block
    int aBegin = K * BLOCK_SIZE * block_i;
 
    // Index of the last sub-matrix of A processed 
    // by the block
    int aEnd   = aBegin + K - 1;
 
    // Step size used to iterate through the 
    // sub-matrices of A
    int aStep  = BLOCK_SIZE;
 
    // Index of the first sub-matrix of B processed 
    // by the block
    int bBegin = BLOCK_SIZE * block_j;
 
    // Step size used to iterate through the 
    // sub-matrices of B
    int bStep  = BLOCK_SIZE * N;
 
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {

        // Declaration of the local memory array As 
        // used to store the sub-matrix of A
        __local int As[BLOCK_SIZE][BLOCK_SIZE];
 
        // Declaration of the local memory array Bs 
        // used to store the sub-matrix of B
        __local int Bs[BLOCK_SIZE][BLOCK_SIZE];
 
        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[thread_i][thread_j] = A[a + K * thread_i + thread_j];
        Bs[thread_i][thread_j] = B[b + N * thread_i + thread_j];
 
        // Synchronize to make sure the matrices 
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            acc += As[thread_i][k] * Bs[k][thread_j];
 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
 
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = N * BLOCK_SIZE * block_i + BLOCK_SIZE * block_j;
    C[c + N * thread_i + thread_j] = acc;

}


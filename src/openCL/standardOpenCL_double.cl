__kernel void mat_mult( __global const double *A, __global const double *B, __global double *C, 
						const int M, const int N, const int K) {
	int i, j, k;
	i = get_global_id(0);
	j = get_global_id(1);
	double acc = 0;	
	
	for (k = 0; k < K; ++k) {
		acc += A[i*K + k] * B[k*N + j];
	}
	
	C[i*N + j] = acc;
}


#include <iostream>
#include <fstream> 
#include <sys/time.h>
#include <sys/resource.h>
#include <iomanip> // Output formatting
#include <immintrin.h>
#include <omp.h>

using namespace std;

const int g_outputWidth = 9; 	  	// Set the output width for elements.
const int g_outputPrecision = 15; 	// Output elements precision.
int g_cacheBlockSize;  				// Block size for loop tilling/blocking.
int g_dataType;


void getTiming(double* timing, struct timeval* time_start, struct timeval* time_end) {
	*timing = 0;

	*timing = (double)(time_end->tv_sec- time_start->tv_sec)*1000;
	*timing += (double)(time_end->tv_usec - time_start->tv_usec)/1000;

	return;				   
}

void inline start_timing(struct rusage* usage, struct timeval* user_start, struct timeval* system_start) {
	getrusage(RUSAGE_SELF, usage); 
	*user_start = usage->ru_utime;
	*system_start = usage->ru_stime;	
}

void inline end_timing(struct rusage* usage, struct timeval* user_end, struct timeval* system_end) {
	getrusage(RUSAGE_SELF, usage); 
	*user_end = usage->ru_utime;
	*system_end = usage->ru_stime;	
}

void getTiming_User_System(double* timing, struct timeval* time_start1, struct timeval* time_end1,
										   struct timeval* time_start2, struct timeval* time_end2) {
	double aux;
	getTiming(&aux, time_start1, time_end1);
	getTiming(timing, time_start2, time_end2);
	*timing += aux;										   
}


// Matrix Multiplication with loop unrolling x8 and blocking.
void matrixMultiplicationOptimized_float(int A_dim1, int A_dim2, int B_dim1, float** A, float** B, float** C){
	int i, j, k, l;
	int limit0 = A_dim1; 			// Index i limit 
	int limit1 = B_dim1; 			// Index j limit
	int limit2 = A_dim2; 			// Index k limit
	int aux_i, aux_j, aux_k;
	int aux_limit_i; 	 			// Block index limit i
	int aux_limit_j; 	 			// Block index limit j
	int aux_limit_k; 	 			// Block index limit k
	int unroll_factor = 8;
	int unroll_limit; 	 			// Loop unroll index limit
	
	#pragma omp parallel for private(i, j, k, aux_i, aux_j, aux_k, aux_limit_i, aux_limit_j, aux_limit_k, unroll_limit) shared (A,B,C)
	for (i = 0; i < limit0; i += g_cacheBlockSize) {
	    // Blocking index i limit
	    aux_limit_i = min((i+g_cacheBlockSize), limit0);
	    
		for (j = 0; j < limit1; j += g_cacheBlockSize) {
		    // Blocking index j limit
		    aux_limit_j = min((j+g_cacheBlockSize), limit1);
		    
			for (k = 0; k < limit2; k += g_cacheBlockSize) {
			    // Blocking index k limit
			    aux_limit_k = min((k+g_cacheBlockSize), limit2);
			    
	            unroll_limit = aux_limit_k - (unroll_factor-1); // Unrolling by factor of 8
	            
	          	for(aux_i = i; aux_i < aux_limit_i; ++aux_i) {
	            	for(aux_j = j; aux_j < aux_limit_j; ++aux_j) {
						float zero = 0;
						__m256 acc = _mm256_broadcast_ss(&zero);	
						
						// Unrolling for k loop
	                	for(aux_k = k; aux_k < unroll_limit; aux_k+=unroll_factor) {	
	                    	acc = _mm256_add_ps(acc, 
	                    		  _mm256_mul_ps( _mm256_load_ps( &A[aux_i][aux_k] ), _mm256_load_ps( &B[aux_j][aux_k] ) ) );
	                    } 
					
	                    // Gather possible uncounted elements
	                    for (; aux_k < aux_limit_k; ++aux_k)
	                    	C[aux_i][aux_j] += A [aux_i][aux_k] * B [aux_j][aux_k];   
	                   	
	                   	
	                    // Sum up everything
	                    float acc_vet[8];
	                    
	                    _mm256_storeu_ps(acc_vet, acc);
	                    C[aux_i][aux_j] += acc_vet[0] + acc_vet[1] + acc_vet[2] + acc_vet[3]
										 + acc_vet[4] + acc_vet[5] + acc_vet[6] + acc_vet[7];
					
	            	}  
	        	}   
			}
		}
	}
	return;
}
	
// Matrix Multiplication with loop unrolling x4 and blocking.
void matrixMultiplicationOptimized_double(int A_dim1, int A_dim2, int B_dim1, double** A, double** B, double** C) {
	int i, j, k, l;
	int limit0 = A_dim1; 			// Index i limit 
	int limit1 = B_dim1; 			// Index j limit
	int limit2 = A_dim2; 			// Index k limit
	int aux_i, aux_j, aux_k;
	int aux_limit_i; 	 			// Block index limit i
	int aux_limit_j; 	 			// Block index limit j
	int aux_limit_k; 	 			// Block index limit k
	int unroll_factor = 4;
	int unroll_limit; 	 			// Loop unroll index limit
	
	#pragma omp parallel for private(i, j, k, aux_i, aux_j, aux_k, aux_limit_i, aux_limit_j, aux_limit_k, unroll_limit) shared (A,B,C)
	for (i = 0; i < limit0; i += g_cacheBlockSize) {
	    // Blocking index i limit
	    aux_limit_i = min((i+g_cacheBlockSize), limit0);
	    
		for (j = 0; j < limit1; j += g_cacheBlockSize) {
		    // Blocking index j limit
		    aux_limit_j = min((j+g_cacheBlockSize), limit1);
		    
			for (k = 0; k < limit2; k += g_cacheBlockSize) {
			    // Blocking index k limit
			    aux_limit_k = min((k+g_cacheBlockSize), limit2);
			    
	            unroll_limit = aux_limit_k - (unroll_factor-1); // Unrolling by factor of 4
	            
	          	for(aux_i = i; aux_i < aux_limit_i; ++aux_i) {
	            	for(aux_j = j; aux_j < aux_limit_j; ++aux_j) {
						
						double zero = 0;	
						__m256d acc = _mm256_broadcast_sd(&zero);
						
						// Unrolling for k loop
	                	for(aux_k = k; aux_k < unroll_limit; aux_k+=unroll_factor) {
	                    	acc = _mm256_add_pd(acc, 
	                    		  _mm256_mul_pd( _mm256_load_pd( &A[aux_i][aux_k] ), _mm256_load_pd( &B[aux_j][aux_k] ) ) );
	                    } 
					
	                    // Gather possible uncounted elements
	                    for (; aux_k < aux_limit_k; ++aux_k)
	                    	C[aux_i][aux_j] += A [aux_i][aux_k] * B [aux_j][aux_k];   
	                   	
	                   	
	                    // Sum up everything
	                    double acc_vet[4];
	                    
						_mm256_storeu_pd(acc_vet, acc);
	                    
	                    C[aux_i][aux_j] += acc_vet[0] + acc_vet[1] + acc_vet[2] + acc_vet[3]; 
					
	            	}  
	        	}   
			}
		}
	}
	return;
}


template<class E> class init
{

public:

	// Read matrices A and B from input text
	int readInput(E** matrix_A, E** matrix_B, int m, int k, int n) {
		ifstream file_matrix_A;
		ifstream file_matrix_B;
		file_matrix_A.open ("matrix_A.txt", std::ifstream::in); //File to read matrix A
		file_matrix_B.open ("matrix_B.txt", std::ifstream::in); //File to read matrix B
	
		int dim1 = m;
		int dim2 = k;

		if (file_matrix_A.is_open()) {
			for (int i = 0; i < dim1; ++i) {
				for (int j = 0; j < dim2; ++j) {
						file_matrix_A >> matrix_A[i][j]; 
				}
			}
		} else return 0;
	
		dim1 = k;
		dim2 = n;
	
		if (file_matrix_B.is_open()) {
			for (int i = 0; i < dim1; ++i) {
				for (int j = 0; j < dim2; ++j) {
						file_matrix_B >> matrix_B[j][i]; 	
				}
			}
		} else return 0;
	
		file_matrix_B.close();

		return 1;
	}

	void writeReadTime(double time){
		ofstream file_out;
		file_out.open("optimizedMultParallel2_read_time.txt");
		if (file_out.is_open()) {
			file_out << time;
			file_out.close();
		}	
	}

	init(int m, int k, int n, void*** A, void*** B, void*** C) {
	
		double read_time;
		double aux;
		struct timeval system_start, system_end, user_start, user_end; // Structs used by rusage
		struct rusage usage;
	
		// Timing start.
		start_timing(&usage, &user_start, &system_start);
		
		E*** matrix_A = (E***) A;
		E*** matrix_B = (E***) B;
		E*** matrix_C = (E***) C;
	
		int i, j;
	
		*matrix_A = (E**) malloc (m * sizeof(E*) );
		*matrix_B = (E**) malloc (n * sizeof(E*) );
		*matrix_C = (E**) malloc (m * sizeof(E*) );

		// Create matrix A
		for (i = 0; i < m; ++i)
			(*matrix_A)[i] = (E*) _mm_malloc (k * sizeof(E), 32);
		
		// Create matrix B (column order major)
		for (i = 0; i < n; ++i)
			(*matrix_B)[i] = (E*) _mm_malloc (k * sizeof(E), 32);
	
		// Fill matrix C with 0
		for (i = 0; i < m; ++i) {
			(*matrix_C)[i] = (E*) _mm_malloc (n * sizeof(E), 32);
			for (j = 0; j < n; ++j)
				(*matrix_C)[i][j] = 0;
		}
	
		// Read Matrices A and B from input
		if (readInput(*matrix_A, *matrix_B, m, k, n) != 1)  // If return is not 1, something is wrong
			return;
	
		// Timing end.
		end_timing(&usage, &user_end, &system_end);
	
		// Get Time.
		getTiming_User_System(&read_time, &user_start, &user_end, &system_start, &system_end);
	
		// Write read time.
		writeReadTime(read_time);
	}

};

int readDimensions(int* m, int* k, int* n, int* cacheBlockSize, int* dataType) {
	ifstream file_input;
	file_input.open ("matrix_dim.txt", std::ifstream::in); //File to read matrices dimensions
	if (file_input.is_open()) {
		file_input >> *m >> *k >> *n >> *cacheBlockSize >> *dataType;
		file_input.close();
	} else return 0;
	return 1;
}

// Outputs the resulting matrix C. C = A*B
template <class E> 
void writeOutput(E** matrix_C, int dim1, int dim2) {
	ofstream file_out;
	file_out.open("optimizedMultParallel2_matrix_C.txt");
	if (file_out.is_open()) {
		file_out << std::fixed;
		for (int i = 0; i < dim1; ++i) {
			for (int j = 0; j < dim2; ++j) {
				file_out << std::setprecision(g_outputPrecision) << std::setw(g_outputWidth) << matrix_C[i][j] << " ";
		  	}
		  	file_out << endl;
		}	
	} 
}

void writeWriteTime(double time){
	ofstream file_out;
	file_out.open("optimizedMultParallel2_write_time.txt");
	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeProcessTime(double time){
	ofstream file_out;
	file_out.open("optimizedMultParallel2_process_time.txt");
	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

int main(int argc, char *argv[]) {	
	int m, k, n, cacheBlockSize, dataType;
	double process_time, write_time;
	double aux;
	struct timeval system_start, system_end, user_start, user_end; // Structs used by rusage
	struct rusage usage;
	struct timeval time_start, time_end; // Structs gettimeofday
	
	if (!readDimensions(&m, &k, &n, &cacheBlockSize, &dataType)) { 
		cout << "Couldn't read in.txt file!" << endl;
		return 0;	
	}	
	
	g_cacheBlockSize = cacheBlockSize;
	g_dataType = dataType;
	
	void** matrix_A;
	void** matrix_B;
	void** matrix_C;
	if (dataType == 0) {
		init<int>(m, k, n, &matrix_A, &matrix_B, &matrix_C);
		
		// Timing start.
		start_timing(&usage, &user_start, &system_start);
		
		//matrixMultiplicationOptimized_int(m, k, n, matrix_A, matrix_B, matrix_C);	
		
		// Timing end.
		end_timing(&usage, &user_end, &system_end);
		
	} else if (dataType == 1) {
		
		init<float>(m, k, n, &matrix_A, &matrix_B, &matrix_C);
		
		// Timing start.
		gettimeofday(&time_start, NULL);
		
		matrixMultiplicationOptimized_float(m, k, n, (float**) matrix_A, (float**) matrix_B, (float**) matrix_C);	

	} else {
		
		init<double>(m, k, n, &matrix_A, &matrix_B, &matrix_C);
		
		// Timing start.
		gettimeofday(&time_start, NULL);
		
		matrixMultiplicationOptimized_double(m, k, n, (double**) matrix_A, (double**) matrix_B, (double**) matrix_C);	

	}	
	
	// Timing end.
	gettimeofday(&time_end, NULL);
    getTiming(&process_time, &time_start, &time_end);
	
	// Write processing time.
	writeProcessTime(process_time);
	

	// Timing start.
	start_timing(&usage, &user_start, &system_start);

	if (g_dataType == 1) {
		writeOutput<float>( (float**) matrix_C, m, n);
	} else {
		writeOutput<double>( (double**) matrix_C, m, n);
	}	
	
	// Timing end.
	end_timing(&usage, &user_end, &system_end);

	// Get Time.
	getTiming_User_System(&write_time, &user_start, &user_end, &system_start, &system_end);

	// Write writing time.
	writeWriteTime(write_time);
			
    return 0;
}




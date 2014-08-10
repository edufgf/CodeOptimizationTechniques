#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include <thread>  					// Using one of C++11 thread functions
#include <iomanip> 					// Output formatting
#include <pthread.h>
#include <immintrin.h>
#include "Array2D.h"

using namespace std;

typedef double E; // Define the type of elements being used.

const int g_maxThreads = 7; 	   // Maximum number of threads this program do use ( dont need more ).
const int g_truncateSize = 128;    // Size of a matrix side when it is small enough to run on standard multiplication.
const int g_outputWidth = 9; 	   // Set the output width for elements.
const int g_outputPrecision = 15;  // Matrix C elements output precision.
int g_cacheBlockSize = 16;   	   // Block size for loop tilling/blocking.

void strassenMultiplication(const Array2D<E> &matrix_A, const Array2D<E> &matrix_B, int ini_i, int ini_j, int size,
                            Array2D<E> &matrix_C, Array2D<E> &matrix_resultA, Array2D<E> &matrix_resultB);

void matrixSub(const Array2D<E> &matrix_A, const Array2D<E> &matrix_B, int ini_i0, int ini_j0, int ini_i1, int ini_j1, 
               int ini_i2, int ini_j2, int size, Array2D<E> &matrix_Result);

void matrixAdd(const Array2D<E> &matrix_A, const Array2D<E> &matrix_B, int ini_i0, int ini_j0, int ini_i1, int ini_j1, 
               int ini_i2, int ini_j2, int size, Array2D<E> &matrix_Result);

void* initWork(void*);

// Class that handle threads and it's attributes.
class Thread	
{
public:
	int m_size;		
	int m_quadrant;  			     // Quadrant which will store results.
	int m_matrix_id; 				 // Which matrix_M is being calculated, M1,M2,M3..M7.
	int m_calculate_quadrant;   	 // If not 0, we will do operation on quadrants, not compute matrix_M.
	int m_started;					 // Thread running flag.
	Array2D<E> *m_p_matrix_resultA;  // Buffer matrix_resultA, each thread has its own.
	Array2D<E> *m_p_matrix_resultB;  // Buffer matrix_resultB, each thread has its own.
	Array2D<E> *m_p_matrix_M;   // M = A*B. Each thread has its own matrix_M.
	Array2D<E> *m_p_matrix_A;   // Matrix_A, each thread has its own copy calculated from calling thread.
	Array2D<E> *m_p_matrix_B;   // Matrix_B, each thread has its own copy calculated from calling thread.
	Array2D<E> *m_p_matrix_C;   // Matrix_C, pointer to the main matrix with final results.
	Array2D<E> *m_p_matrix_M4;  // Matrix_M4, pointer to the auxiliar M4 from calling thread.
	Array2D<E> *m_p_matrix_M5;  // Matrix_M5, pointer to the auxiliar M5 from calling thread.
	Array2D<E> *m_p_matrix_M7;  // Matrix_M7, pointer to the auxiliar M7 from calling thread.
	int *m_p_finished_M;        // Flag array finished_M, if M[i] was calculated.
	int *m_p_finished_quadrant; // Flag array finished_quadrant, if quadrant[i] was calculated.
	int quadrant_point[4][2];   // Auxiliar array for starting index for each quadrant.
	pthread_t m_pt;
	
	
	Thread() { m_started = 0; }

	~Thread() { }

	bool started() const {
		return m_started;
	}
	
	void start(){
		m_started = 1;
		pthread_create(&m_pt, NULL, initWork, (void*)this);
	}


	void join(){
		if(m_pt != 0 && m_started){
			pthread_join(m_pt,NULL);
			m_started = 0;		
		}
	}
	
    void initThread(Array2D<E> *m_matrix_C,Array2D<E> *m_matrix_M4, Array2D<E> *m_matrix_M5, Array2D<E> *m_matrix_M7, 
                    int size, int *finished_M, int *finished_quadrant) {
    	m_p_matrix_resultA = new Array2D<E>(size/2, size/2, 1, 0); // Buffer matrix to store additions, subtractions and copies
    	m_p_matrix_resultB = new Array2D<E>(size/2, size/2, 1, 0); // Buffer matrix to store additions, subtractions and copies
    	m_p_matrix_M = new Array2D<E>(size, size, 1); 	         // Auxiliar matrix to store result C from A*B
    	m_p_matrix_A = new Array2D<E>(0, 0, 1); 		   // Auxiliar matrix to store A
    	m_p_matrix_B = new Array2D<E>(0, 0, 1); 		   // Auxiliar matrix to store B
    	m_p_matrix_C = m_matrix_C;   			   // Pointer to matrix_C
    	m_p_matrix_M4 = m_matrix_M4; 			   // Pointer to M4
    	m_p_matrix_M5 = m_matrix_M5; 			   // Pointer to M5				
        m_p_matrix_M7 = m_matrix_M7; 			   // Pointer to M7
    	m_size = size;               			   // This thread matrix_M size
    	m_p_finished_M = finished_M;			   // Copying finished_M array flag
    	m_p_finished_quadrant = finished_quadrant; // Copying finished_quadrant array flag
    	m_calculate_quadrant = 0;  				   // Default, this thread should not calculate a quadrant
    	
    	// Auxiliar starting index for each quadrant
    	quadrant_point[0][0] = 0;   
    	quadrant_point[0][1] = 0;
    	quadrant_point[1][0] = size;
    	quadrant_point[1][1] = 0;
    	quadrant_point[2][0] = 0;
    	quadrant_point[2][1] = size;
    	quadrant_point[3][0] = size;
    	quadrant_point[3][1] = size;
    }
     
    void doWork() {
        if (m_calculate_quadrant == 0) { // Calculate matrix_M not quadrant
		    (*m_p_matrix_M).fill(0); 	 // Refill this thread auxiliar matrix_M for reuse
		    
		    // M = A * B and use matrix_result as buffers for additions, subtractions and copies
			strassenMultiplication(*m_p_matrix_A, *m_p_matrix_B, 0, 0, m_size, *m_p_matrix_M, *m_p_matrix_resultA, 
									*m_p_matrix_resultB);	
									
		    if (m_quadrant == -1) {  				  // Write to one of the buffers M4, M5 or M7
		    	if (m_matrix_id == 4) {
		    		*m_p_matrix_M4 = *(m_p_matrix_M); // Copying result to M4
		    	} else if (m_matrix_id == 5) {
		    	    *m_p_matrix_M5 = *(m_p_matrix_M); // Copying result to M5
		    	} else if (m_matrix_id == 7) {
		    	    *m_p_matrix_M7 = *(m_p_matrix_M); // Copying result to M7
		    	}
		    } else { 										   // This case we write directly to the matrix_C
				int aux_ini_i = quadrant_point[m_quadrant][0]; // auxiliar starting index i in this quadrant
				int aux_ini_j = quadrant_point[m_quadrant][1]; // auxiliar starting index j in this quadrant
				
				// Store result in matrix_C
				matrixAdd(*m_p_matrix_C, *(m_p_matrix_M), aux_ini_i, aux_ini_j, 0, 0, aux_ini_i, aux_ini_j, m_size, 	
				          *m_p_matrix_C); 
    		}
		    m_p_finished_M[m_matrix_id] = 1; // Set flag that this matrix_M was computed
		    
        } else if (m_calculate_quadrant == 1) {
            // Calculate partially C11 = M1 + M7 
        	matrixAdd(*m_p_matrix_C, *m_p_matrix_M7, 0, 0, 0, 0, 0, 0, m_size, *m_p_matrix_C);
        	m_p_finished_quadrant[m_calculate_quadrant] = 1;  // Set flag that this quadrant was computed
		} else if (m_calculate_quadrant == 2) {
			// Calculate C21 = M2 + M4
			matrixAdd(*m_p_matrix_C, *m_p_matrix_M4, m_size, 0, 0, 0, m_size, 0, m_size, *m_p_matrix_C);
			m_p_finished_quadrant[m_calculate_quadrant] = 1;  // Set flag that this quadrant was computed
		} else if (m_calculate_quadrant == 3) {
			// Calculate C12 = M3 + M5
			matrixAdd(*m_p_matrix_C, *m_p_matrix_M5, 0, m_size, 0, 0, 0, m_size, m_size, *m_p_matrix_C);
			m_p_finished_quadrant[m_calculate_quadrant] = 1;  // Set flag that this quadrant was computed	
		} else if (m_calculate_quadrant == 4) {
			// Calculate C22 = M1 - M2 + M3 + M6
        	matrixAdd(*m_p_matrix_C, *m_p_matrix_C, m_size, m_size, 0, 0, m_size, m_size, m_size, *m_p_matrix_C);      // C22 = M1 + M6
			matrixAdd(*m_p_matrix_C, *m_p_matrix_C, 0, m_size, m_size, m_size, m_size, m_size, m_size, *m_p_matrix_C); // C22 = M3 + C22
			matrixSub(*m_p_matrix_C, *m_p_matrix_C, m_size, m_size, m_size, 0, m_size, m_size, m_size, *m_p_matrix_C); // C22 = C22 - M2
			m_p_finished_quadrant[m_calculate_quadrant] = 1;  // Set flag that this quadrant was computed
		}
	}
	
};

void* initWork(void* arg){
	Thread* my_thread = (Thread *) arg;
	my_thread->doWork();
	my_thread->m_started = 0;
}

// Calculates matrix multiplication between matrix_A and matrix_B. Store the result in matrix_C.
// ini_i means starting index i and ini_j means starting index j.
// Looping through lines ini_i <= i <= ini_i+size and columns ini_j <= j <= ini_j+size from A and B
// Loop unroll and loop blocking applied for performance purposes
void matrixMultiplication(int ini_i, int ini_j, int size, const Array2D<E> &A, const Array2D<E> &B, Array2D<E> &C){
	int i, j, k, l;
	int limit0 = size + ini_i; // Index i limit 
	int limit1 = size + ini_j; // Index j limit
	int limit2 = size;         // Index k limit
    int aux_i, aux_j, aux_k;
    int aux_limit_i; 		   // Block index limit i
    int aux_limit_j; 		   // Block index limit j
    int aux_limit_k; 		   // Block index limit k
    int unroll_factor = 4;
    int unroll_limit; 		        // Loop unroll index limit
    double zero = 0;				// To initialize register ymm

    for (i = ini_i; i < limit0; i += g_cacheBlockSize) {
        // Blocking index i limit
        aux_limit_i = min((i+g_cacheBlockSize), limit0);
        
    	for (j = ini_j; j < limit1; j += g_cacheBlockSize) {
    	    // Blocking index j limit
    	    aux_limit_j = min((j+g_cacheBlockSize), limit1);
    	    
    		for (k = 0; k < limit2; k += g_cacheBlockSize) {
    		    // Blocking index k limit
    		    aux_limit_k = min((k+g_cacheBlockSize), limit2);
    		    
                unroll_limit = aux_limit_k - (unroll_factor-1); // Unrolling by factor of 4
                
              	for(aux_i = i; aux_i < aux_limit_i; ++aux_i) {
                	for(aux_j = j; aux_j < aux_limit_j; ++aux_j) {
						
                    	__m256d acc = _mm256_broadcast_sd(&zero);
                    	
                    	int endA = aux_i*A.getCollumnSize() + ini_j;
                    	int endB = aux_j*B.getCollumnSize() + ini_i;
						
						// Unrolling for k loop
                    	for(aux_k = k; aux_k < unroll_limit; aux_k+=unroll_factor) {
                        	acc = _mm256_add_pd(acc, 
                        		  _mm256_mul_pd( _mm256_load_pd( A.getAddress(endA + aux_k) ), 
                        		  			     _mm256_load_pd( B.getAddress(endB + aux_k) ) ) );
                        } 
                        
                        int endC = aux_i*C.getCollumnSize() + aux_j;
                        
                        // Gather possible uncounted elements
                        for (; aux_k < aux_limit_k; ++aux_k)
                        	C(endC) += A (endA + aux_k) * B (aux_j, aux_k+ini_i);   
                       
                        // Sum up everything
                        E acc_vet[4];
                        
                        _mm256_storeu_pd(acc_vet, acc);
                        
                        C(endC) += acc_vet[0] + acc_vet[1] + acc_vet[2] + acc_vet[3];

                	}  
            	}   
			}
		}
	}
    return;
}

// Copy size*size elements from matrix_A to matrix_Result. 
// Starting indexes are ini_i0 (lines) and ini_j0 (columns) for matrix_A. The same follows for ini_i1 and ini_j1 for matrix_Result 
void matrixCopy(const Array2D<E> &matrix_A, int ini_i0, int ini_j0, int ini_i1, int ini_j1, int size, Array2D<E> &matrix_Result) {  
	for (int i = 0; i < size; ++i){
    	for (int j = 0; j < size; ++j){
    		matrix_Result(i+ini_i1 , j+ini_j1) = matrix_A(i+ini_i0 , j+ini_j0);
    	}
	}			      
	return;		   
}

// Subtract matrices A and B, and store result in matrix_Result.
// Starting indexes are ini_i0 (lines), ini_j0 (columns) for matrix_A. ini_i1, ini_j1 for matrix_B and ini_i2, ini_j2 for matrix_Result 
void matrixSub(const Array2D<E> &matrix_A, const Array2D<E> &matrix_B, int ini_i0, int ini_j0, int ini_i1, int ini_j1, 
               int ini_i2, int ini_j2, int size, Array2D<E> &matrix_Result) {
	for (int i = 0; i < size; ++i){
    	for (int j = 0; j < size; ++j){
    		matrix_Result(i+ini_i2 , j+ini_j2) = matrix_A(i+ini_i0 , j+ini_j0) - matrix_B(i+ini_i1 , j+ini_j1);
    	}
	}	   		   
	return;		   
}

// Add matrices A and B, and store result in matrix_Result.
// Starting indexes are ini_i0 (lines), ini_j0 (columns) for matrix_A. ini_i1, ini_j1 for matrix_B and ini_i2, ini_j2 for matrix_Result 
void matrixAdd(const Array2D<E> &matrix_A, const Array2D<E> &matrix_B, int ini_i0, int ini_j0, int ini_i1, int ini_j1, 
               int ini_i2, int ini_j2, int size, Array2D<E> &matrix_Result) {
	for (int i = 0; i < size; ++i){
    	for (int j = 0; j < size; ++j){
    		matrix_Result(i+ini_i2 , j+ini_j2) = matrix_A(i+ini_i0 , j+ini_j0) + matrix_B(i+ini_i1 , j+ini_j1);
    	}
	}		
	return;		   
}	

// Calculates Strassen multiplication within the range: lines ini_i <= i <= ini_i+size and columns ini_j <= j <= ini_j+size
void strassenMultiplication(const Array2D<E> &matrix_A, const Array2D<E> &matrix_B, int ini_i, int ini_j, int size,
                            Array2D<E> &matrix_C, Array2D<E> &matrix_resultA, Array2D<E> &matrix_resultB) {
    if (size <= g_truncateSize){ // Small enough to run standard multiplication
    	matrixMultiplication(ini_i, ini_j, size, matrix_A, matrix_B, matrix_C);
    } else {
        int half_size = size/2;		   // Size for each four quadrant
        int mid_i = ini_i + half_size; // Index middle of side1
        int mid_j = ini_j + half_size; // Index middle of side2
        int end_i = ini_i + size;	   // Index end limit of side1
        int end_j = ini_j + size; 	   // Index end limit of side2	

        // Matrix A quadrants
        Array2D<E> submatrix_A11(half_size, half_size, 1);
        Array2D<E> submatrix_A12(half_size, half_size, 1);
        Array2D<E> submatrix_A21(half_size, half_size, 1);
        Array2D<E> submatrix_A22(half_size, half_size, 1);
      
        // Matrix B quadrants
        Array2D<E> submatrix_B11(half_size, half_size, 1);
        Array2D<E> submatrix_B12(half_size, half_size, 1);
        Array2D<E> submatrix_B21(half_size, half_size, 1);
        Array2D<E> submatrix_B22(half_size, half_size, 1);
        
        // Matrices M_i are calculated and stored directly into matrix_C.
		// We save up some memory by using the same matrix_M1 as a buffer for each
		// computation and then filling it with zeros for the next calculation
		// Two auxiliar matrices are needed to hold M4 and M5 values
        Array2D<E> matrix_M1(half_size, half_size, 1, 0);
        Array2D<E> matrix_M2(half_size, half_size, 1, 0);

		// Getting submatrices from A and B
		for (int i = 0; i < half_size; ++i) {
			for (int j = 0; j < half_size; ++j) {
				submatrix_A11(i, j) = matrix_A(i, j);
				submatrix_A12(i, j) = matrix_A(i, j+half_size);
				submatrix_A21(i, j) = matrix_A(i+half_size, j); 
				submatrix_A22(i, j) = matrix_A(i+half_size, j+half_size);

				submatrix_B11(j, i) = matrix_B(j, i);
				submatrix_B12(j, i) = matrix_B(j+half_size, i);
				submatrix_B21(j, i) = matrix_B(j, i+half_size); 
				submatrix_B22(j, i) = matrix_B(j+half_size, i+half_size);
			}
		}

        // Quadrants order
		// C11 - 1 | 3 - C12
		// 		 -----
		// C21 - 2 | 4 - C22
    
		// Calculating M1 = (A11 + A22) * (B11 + B22)
        matrixAdd(submatrix_A11, submatrix_A22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultA);
        matrixAdd(submatrix_B11, submatrix_B22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultB);
        strassenMultiplication(matrix_resultA, matrix_resultB, 0, 0, half_size, matrix_M1, matrix_resultA, matrix_resultB);
        
        // Copying directly to matrix_C (answer matrix)
        matrixCopy(matrix_M1, 0, 0, 0, 0, half_size, matrix_C);
        matrix_M1.fill(0); // Filling with zeros for reuse
        
        // Calculating M2 = (A21 + A22) * (B11)
        matrixAdd(submatrix_A21, submatrix_A22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultA);
        matrixCopy(submatrix_B11, 0, 0, 0, 0, half_size, matrix_resultB);
        strassenMultiplication(matrix_resultA, matrix_resultB, 0, 0, half_size, matrix_M1, matrix_resultA, matrix_resultB);
        
        matrixCopy(matrix_M1, 0, 0, mid_i, 0, half_size, matrix_C);
        matrix_M1.fill(0);
        
        // Calculating M3 = (A11) * (B12 - B22)
        matrixCopy(submatrix_A11, 0, 0, 0, 0, half_size, matrix_resultA);
        matrixSub(submatrix_B12, submatrix_B22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultB);
        strassenMultiplication(matrix_resultA, matrix_resultB, 0, 0, half_size, matrix_M1, matrix_resultA, matrix_resultB);
        
        matrixCopy(matrix_M1, 0, 0, 0, mid_j, half_size, matrix_C);
        matrix_M1.fill(0);
        
        // Calculating M6 = (A21 - A11) * (B11 + B12)
        matrixSub(submatrix_A21, submatrix_A11, 0, 0, 0, 0, 0, 0, half_size, matrix_resultA);
        matrixAdd(submatrix_B11, submatrix_B12, 0, 0, 0, 0, 0, 0, half_size, matrix_resultB);
        strassenMultiplication(matrix_resultA, matrix_resultB, 0, 0, half_size, matrix_M1, matrix_resultA, matrix_resultB);
        
        matrixCopy(matrix_M1, 0, 0, mid_i, mid_j, half_size, matrix_C);
        matrix_M1.fill(0);
        
        // Calculate C22 = M1 - M2 + M3 + M6
        matrixAdd(matrix_C, matrix_C, mid_i, mid_j, 0, 0, mid_i, mid_j, half_size, matrix_C);     // C22 = M1 + M6
        matrixAdd(matrix_C, matrix_C, 0, mid_j, mid_i, mid_j, mid_i, mid_j, half_size, matrix_C); // C22 = M3 + C22
        matrixSub(matrix_C, matrix_C, mid_i, mid_j, mid_i, 0, mid_i, mid_j, half_size, matrix_C); // C22 = C22 - M2
          
        // Calculating M4 = (A22) * (B21 - B11)
        matrixCopy(submatrix_A22, 0, 0, 0, 0, half_size, matrix_resultA);
        matrixSub(submatrix_B21, submatrix_B11, 0, 0, 0, 0, 0, 0, half_size, matrix_resultB);
        strassenMultiplication(matrix_resultA, matrix_resultB, 0, 0, half_size, matrix_M1, matrix_resultA, matrix_resultB);
        
        // Calculate C21 = M2 + M4
        matrixAdd(matrix_C, matrix_M1, mid_i, 0, 0, 0, mid_i, 0, half_size, matrix_C);
        
		// Not resetting auxiliar matrix_M1 ! Holding M4 value...

        // Calculating M5 = (A11 + A12) * (B22)
        matrixAdd(submatrix_A11, submatrix_A12, 0, 0, 0, 0, 0, 0, half_size, matrix_resultA);
        matrixCopy(submatrix_B22, 0, 0, 0, 0, half_size, matrix_resultB);
        strassenMultiplication(matrix_resultA, matrix_resultB, 0, 0, half_size, matrix_M2, matrix_resultA, matrix_resultB);   
       
        // Calculate C12 = M3 + M5
        matrixAdd(matrix_C, matrix_M2, 0, mid_j, 0, 0, 0, mid_j, half_size, matrix_C);
        
        // auxiliar matrix_M1 = M4, auxiliar matrix_M2 = M5 
        // M4 = M4 - M5
        matrixSub(matrix_M1, matrix_M2, 0, 0, 0, 0, 0, 0, half_size, matrix_M1);
        
        // Calculating partially C11 = M1 + M4 - M5
        matrixAdd(matrix_C, matrix_M1, 0, 0, 0, 0, 0, 0, half_size, matrix_C);
        
        matrix_M1.fill(0);
        
        // Calculating M7 = (A12 - A22) * (B21 + B22)
        matrixSub(submatrix_A12, submatrix_A22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultA);
        matrixAdd(submatrix_B21, submatrix_B22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultB);
        strassenMultiplication(matrix_resultA, matrix_resultB, 0, 0, half_size, matrix_M1, matrix_resultA, matrix_resultB);
        
        // Calculate C11 = M1 + M4 - M5 + M7
        matrixAdd(matrix_C, matrix_M1, 0, 0, 0, 0, 0, 0, half_size, matrix_C);

    }
    return;
}

// Manage multiple threads to calculate Strassen multiplication of C = A*B. Size = matrix side size.
void parallelStrassenMultiplication(const Array2D<E> &matrix_A, const Array2D<E> &matrix_B, int numThreads, int size,
								    Array2D<E> *matrix_C, Thread *thread) {
	// The order in which matrices M are calculated are done in a way to avoid waiting for other threads
	// This order tries to keep free threads busy as soon as it is possible
	
	int nextThread = 0;
	int finished_M[8]; 			  // Flag to indicate if Matrix_M[1..7] was computed
	int finished_quadrant[5]; 	  // Flag to indicate if quadrant[1..4] was computed
	int thread_id_M[8];		  	  // Thread ID responsible for calculating Matrix_M[1..7] 
	int thread_id_quadrant[5]; 	  // Thread ID responsible for calculating quadrant[1..4]
	int sleep_milliseconds = 100; // Time to sleep when waiting for other threads to finish
	
	// Nothing calculated at the beginning
	for (int i = 0; i < 8; ++i)
		finished_M[i] = 0;
	for (int i =0; i < 5; ++i)
	    finished_quadrant[i] = 0;
	
	int half_size = size/2; // Size for each four quadrant 
	int mid_i = half_size;  // Index middle of side1
    int mid_j = half_size;  // Index middle of side2
	
	// Matrices M1, M2, M3 and M4 are written directly onto the final matrix C
	// Saving up some memory, we only need auxiliar space to store M4, M5 and M6
	Array2D<E> matrix_M4(half_size, half_size, 1); // Half_size is the size of a quadrant
	Array2D<E> matrix_M5(half_size, half_size, 1);
	Array2D<E> matrix_M7(half_size, half_size, 1);
	
	// Initialize each thread passing matrices C, M4, M5 and M7 as pointers.
	// As well passing the flag arrays finished_M and finished_quadrant
	for (int i = 0; i < numThreads; ++i){
	    thread[i].initThread(matrix_C, &matrix_M4, &matrix_M5, &matrix_M7, half_size, &finished_M[0], &finished_quadrant[0]);
	}
	
	// Matrix to store additions, subtractions and copies.
	Array2D<E> matrix_resultA(half_size, half_size, 1);
	Array2D<E> matrix_resultB(half_size, half_size, 1);
	
	// Matrix A quadrants
	Array2D<E> submatrix_A11(half_size, half_size, 1);
    Array2D<E> submatrix_A12(half_size, half_size, 1);
    Array2D<E> submatrix_A21(half_size, half_size, 1);
    Array2D<E> submatrix_A22(half_size, half_size, 1);
    
    // Matrix B quadrants    
    Array2D<E> submatrix_B11(half_size, half_size, 1);
    Array2D<E> submatrix_B12(half_size, half_size, 1);
    Array2D<E> submatrix_B21(half_size, half_size, 1);
    Array2D<E> submatrix_B22(half_size, half_size, 1);
	
	// Getting submatrices from A and B
    for (int i = 0; i < half_size; ++i) {
		for (int j = 0; j < half_size; ++j) {
			submatrix_A11(i, j) = matrix_A(i, j);
			submatrix_A12(i, j) = matrix_A(i, j+half_size);
			submatrix_A21(i, j) = matrix_A(i+half_size, j); 
			submatrix_A22(i, j) = matrix_A(i+half_size, j+half_size);

			submatrix_B11(j, i) = matrix_B(j, i);
			submatrix_B12(j, i) = matrix_B(j+half_size, i);
			submatrix_B21(j, i) = matrix_B(j, i+half_size); 
			submatrix_B22(j, i) = matrix_B(j+half_size, i+half_size);
		}
	}
    
    //Quadrants order
    // C11 - 1 | 3 - C12
	// 		 -----
	// C21 - 2 | 4 - C22
    
	// Calculating M1 = (A11 + A22) * (B11 + B22)
    matrixAdd(submatrix_A11, submatrix_A22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultA);
    matrixAdd(submatrix_B11, submatrix_B22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultB);
    
    // Copy results to thread own result memory
    *(thread[nextThread].m_p_matrix_A) = matrix_resultA;
    *(thread[nextThread].m_p_matrix_B) = matrix_resultB;
    thread[nextThread].m_quadrant = 0;  // M1 stores into upper-left quadrant, 0-indexed
    thread[nextThread].m_matrix_id = 1; // That is M'1'
    thread_id_M[1] = nextThread;        // Responsible thread ID for M1 computation
    thread[nextThread].start();
    
    nextThread = (nextThread+1) % numThreads;
    // If thread has started, wait for it to finish
    if (thread[nextThread].started()) {
    	thread[nextThread].join();
    }
    
    // Calculating M2 = (A21 + A22) * (B11)
	matrixAdd(submatrix_A21, submatrix_A22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultA);
	matrixCopy(submatrix_B11, 0, 0, 0, 0, half_size, matrix_resultB);
	
	*(thread[nextThread].m_p_matrix_A) = matrix_resultA;
	*(thread[nextThread].m_p_matrix_B) = matrix_resultB;
	
	thread[nextThread].m_quadrant = 1;
	thread[nextThread].m_matrix_id = 2;
	thread_id_M[2] = nextThread;
	thread[nextThread].start();
	
    nextThread = (nextThread+1) % numThreads;
    if (thread[nextThread].started()) {
    	thread[nextThread].join();
    }
    
    // Calculating M3 = (A11) * (B12 - B22)
	matrixCopy(submatrix_A11, 0, 0, 0, 0, half_size, matrix_resultA);
    matrixSub(submatrix_B12, submatrix_B22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultB);
    
    *(thread[nextThread].m_p_matrix_A) = matrix_resultA;
	*(thread[nextThread].m_p_matrix_B) = matrix_resultB;
	thread[nextThread].m_quadrant = 2;
	thread[nextThread].m_matrix_id = 3;
	thread_id_M[3] = nextThread;
	thread[nextThread].start();
    
    nextThread = (nextThread+1) % numThreads;
    if (thread[nextThread].started()) {
    	thread[nextThread].join();
    }
    
    // Calculating M6 = (A21 - A11) * (B11 + B12)
    matrixSub(submatrix_A21, submatrix_A11, 0, 0, 0, 0, 0, 0, half_size, matrix_resultA);
    matrixAdd(submatrix_B11, submatrix_B12, 0, 0, 0, 0, 0, 0, half_size, matrix_resultB);
    
    *(thread[nextThread].m_p_matrix_A) = matrix_resultA;
	*(thread[nextThread].m_p_matrix_B) = matrix_resultB;
	thread[nextThread].m_quadrant = 3;
	thread[nextThread].m_matrix_id = 6;
	thread_id_M[6] = nextThread;
	thread[nextThread].start();
    
    nextThread = (nextThread+1) % numThreads;
    if (thread[nextThread].started()) {
    	thread[nextThread].join();
    }
    
    // Calculating M4 = (A22) * (B21 - B11)
    matrixCopy(submatrix_A22, 0, 0, 0, 0, half_size, matrix_resultA);
    matrixSub(submatrix_B21, submatrix_B11, 0, 0, 0, 0, 0, 0, half_size, matrix_resultB);
    
    *(thread[nextThread].m_p_matrix_A) = matrix_resultA;
	*(thread[nextThread].m_p_matrix_B) = matrix_resultB;
	thread[nextThread].m_quadrant = -1; // Won't write into any quadrant, but in auxiliar M4 matrix
	thread[nextThread].m_matrix_id = 4;
	thread_id_M[4] = nextThread;
	thread[nextThread].start();
    
    nextThread = (nextThread+1) % numThreads;
    if (thread[nextThread].started()) {
    	thread[nextThread].join();
    }
    
    // Calculating M5 = (A11 + A12) * (B22)
    matrixAdd(submatrix_A11, submatrix_A12, 0, 0, 0, 0, 0, 0, half_size, matrix_resultA);
    matrixCopy(submatrix_B22, 0, 0, 0, 0, half_size, matrix_resultB);
    
    *(thread[nextThread].m_p_matrix_A) = matrix_resultA;
	*(thread[nextThread].m_p_matrix_B) = matrix_resultB;
	thread[nextThread].m_quadrant = -1; // Won't write into any quadrant, but in auxiliar M5 matrix
	thread[nextThread].m_matrix_id = 5;
	thread_id_M[5] = nextThread;
	thread[nextThread].start();
    
    nextThread = (nextThread+1) % numThreads;
    if (thread[nextThread].started()) {
    	thread[nextThread].join();
    }
    
    // Calculating M7 = (A12 - A22) * (B21 + B22)
    matrixSub(submatrix_A12, submatrix_A22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultA);
    matrixAdd(submatrix_B21, submatrix_B22, 0, 0, 0, 0, 0, 0, half_size, matrix_resultB);
    
    *(thread[nextThread].m_p_matrix_A) = matrix_resultA;
	*(thread[nextThread].m_p_matrix_B) = matrix_resultB;
	thread[nextThread].m_quadrant = -1; // Won't write into any quadrant, but in auxiliar M7 matrix
	thread[nextThread].m_matrix_id = 7;
	thread_id_M[7] = nextThread;
	thread[nextThread].start();
	
	nextThread = (nextThread+1) % numThreads;
    if (thread[nextThread].started()) {
    	thread[nextThread].join();
    }
    
    // M1, M2, M3, M6 were the first calculated, and are the only matrices needed to calculate quadrant 3 (C22)
    if ( !(finished_M[1]) )
    	thread[thread_id_M[1]].join();
    if ( !(finished_M[2]) )
    	thread[thread_id_M[2]].join();
    if ( !(finished_M[3]) )
    	thread[thread_id_M[3]].join();
    if ( !(finished_M[6]) )
    	thread[thread_id_M[6]].join();
	
	// Calculate C22 = M1 - M2 + M3 + M6
	thread[nextThread].m_calculate_quadrant = 4; // Calculate_quadrant is 1-indexed
	thread_id_quadrant[4] = nextThread;
	thread[nextThread].start(); 
	
	nextThread = (nextThread+1) % numThreads;
    if (thread[nextThread].started()) {
    	thread[nextThread].join();
    }
	
	// To compute C21 just need auxiliar matrix M4. M2 is stored in C21 space
	// After computing quadrant 4, we can compute C21 (we don't need to preserve only M2 in C21 anymore)
	if ( !(finished_M[4]) )
    	thread[thread_id_M[4]].join();
    if ( !(finished_quadrant[4]) )
    	thread[thread_id_quadrant[4]].join();

	//Calculate C21 = M2 + M4
	thread[nextThread].m_calculate_quadrant = 2;
	thread_id_quadrant[2] = nextThread;
	thread[nextThread].start();
	
	nextThread = (nextThread+1) % numThreads;
    if (thread[nextThread].started()) {
    	thread[nextThread].join();
    }
	
	// To compute C12 just need auxiliar matrix M5. M3 is stored in C12 space
	// After computing quadrant 4, we can compute C12 (we don't need to preserve only M3 in C12 anymore)
	// Quadrant 4 already finished!
	if ( !(finished_M[5]) )
    	thread[thread_id_M[5]].join();
	
	//Calculate C12 = M3 + M5
	thread[nextThread].m_calculate_quadrant = 3;
	thread_id_quadrant[3] = nextThread;
	thread[nextThread].start();
	
	nextThread = (nextThread+1) % numThreads;
    if (thread[nextThread].started()) {
    	thread[nextThread].join();
    }
	
	// To compute part of C11 just need auxiliar matrix M7. M1 is stored in C11 space
	// After computing quadrant 4, we can compute C11 (we don't need to preserve only M1 in C11 anymore)
	// Quadrant 4 already finished!
	if ( !(finished_M[7]) )
    	thread[thread_id_M[7]].join();
	
	// Calculate part of C11 = M1 + M7
	thread[nextThread].m_calculate_quadrant = 1;
	thread_id_quadrant[1] = nextThread;
	thread[nextThread].start();
	
	// C11 is the only quadrant left to compute. We need to compute M4 - M5
	// We can overwrite M4 value after we have used it to compute C21
	// M5 already finished!
	if ( !(finished_quadrant[2]) )
    	thread[thread_id_quadrant[2]].join();
	
	// Calculate M4 = M4 - M5
	matrixSub(matrix_M4, matrix_M5, 0, 0, 0, 0, 0, 0, half_size, matrix_M4);
	
	// Wait for quadrant one be partially computed, this is to avoid data racing
	// Then compute finally C11
	if ( !(finished_quadrant[1]) )
    	thread[thread_id_quadrant[1]].join();
	
	// Calculate C11 = M1 + M4 - M5 + M7
    matrixAdd(*matrix_C, matrix_M4, 0, 0, 0, 0, 0, 0, half_size, *matrix_C);
	
    // Join all threads and terminate the function
    for (int i = 0; i < numThreads; ++i) {
		thread[i].join();
    }   
}

// Corner case when n = 0 or n = 1
void handleCornerCase(const Array2D<E> &matrix_A, const Array2D<E> &matrix_B, Array2D<E> &matrix_C, int size) {
	if ( size == 1 ) {
	   matrix_C(0, 0) = matrix_A(0 , 0) * matrix_B(0, 0);
	} else if ( size == 2 ) { 
	   // This size would crash multi core forced solution.
	   matrixMultiplication(0, 0, size, matrix_A, matrix_B, matrix_C);
	}
}

// Read matrices A and B from input text
int readInput(Array2D<E> &matrix_A, Array2D<E> &matrix_B, int m, int k, int n) {
	ifstream file_matrix_A;
	ifstream file_matrix_B;
	file_matrix_A.open ("matrix_A.txt", std::ifstream::in); //File to read matrix A
	file_matrix_B.open ("matrix_B.txt", std::ifstream::in); //File to read matrix B
	
	int dim1 = m;
	int dim2 = k;

	if (file_matrix_A.is_open()) {
		for (int i = 0; i < dim1; ++i) {
			for (int j = 0; j < dim2; ++j) {
					file_matrix_A >> matrix_A(i,j); 	
			}
		}
	} else return 0;
	
	dim1 = k;
	dim2 = n;
	
	if (file_matrix_B.is_open()) {
		for (int i = 0; i < dim1; ++i) {
			for (int j = 0; j < dim2; ++j) {
					file_matrix_B >> matrix_B(j,i); 	
			}
		}
	} else return 0;
	
	file_matrix_B.close();

	return 1;
}

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
void writeOutput(Array2D<E> &matrix_C, int dim1, int dim2) {
	ofstream file_out;
	#ifdef FORCE_SINGLE_CORE
		file_out.open("strassenMult2_matrix_C.txt");
	#else
		file_out.open("strassenMultParallel2_matrix_C.txt");	
	#endif
		
	if (file_out.is_open()) {
		file_out << std::fixed;
		for (int i = 0; i < dim1; ++i) {
			for (int j = 0; j < dim2; ++j) {
				file_out << std::setprecision(g_outputPrecision) << std::setw(g_outputWidth) << matrix_C(i,j) << " ";
		  	}
		  	file_out << endl;
		}	
	} 
}

void getTiming(double* timing, struct timeval* time_start, struct timeval* time_end) {
	*timing = 0;
	
	*timing = (double)(time_end->tv_sec- time_start->tv_sec)*1000;
	*timing += (double)(time_end->tv_usec - time_start->tv_usec)/1000;
	
	return;				   
}

void writeReadTime(double time){
	ofstream file_out;
	#ifdef FORCE_SINGLE_CORE
		file_out.open("strassenMult2_read_time.txt");
	#else
		file_out.open("strassenMultParallel2_read_time.txt");
	#endif
	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeWriteTime(double time){
	ofstream file_out;
	#ifdef FORCE_SINGLE_CORE
		file_out.open("strassenMult2_write_time.txt");
	#else
		file_out.open("strassenMultParallel2_write_time.txt");
	#endif
	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeProcessTime(double time){
	ofstream file_out;
	#ifdef FORCE_SINGLE_CORE
		file_out.open("strassenMult2_process_time.txt");
	#else
		file_out.open("strassenMultParallel2_process_time.txt");
	#endif
	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
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


int main() {	
    int m, k, n, cacheBlockSize, dataType;
	
	if (!readDimensions(&m, &k, &n, &cacheBlockSize, &dataType)) { 
		cout << "Couldn't read in.txt file!" << endl;
		return 0;	
	}	
	g_cacheBlockSize = cacheBlockSize;
    
    double read_time, process_time, write_time;
	struct timeval system_start, system_end, user_start, user_end; // Structs used by rusage
	struct rusage usage; 
	struct timeval time_start, time_end; // Structs gettimeofday
    
    int numThreads;		 
    int single_core = 0; 							    // Single core execution 
    int numCores = std::thread::hardware_concurrency(); // Get number of cores
	
	// Timing start.
	start_timing(&usage, &user_start, &system_start);
		
	// Get matrices dimension.
    int size = n;
	
	// Create
	Array2D <E> matrix_A(size, size, 1);
	Array2D <E> matrix_B(size, size, 1);
	Array2D <E> matrix_C(size, size, 1, 0); // Fill with zeros.

    // Read Matrices A and B from input
    if (readInput(matrix_A, matrix_B, size, size, size) != 1)  // If return is not 1, something is wrong
		return 0;
	
	// Timing end.
	end_timing(&usage, &user_end, &system_end);
	
	// Get Time.
	getTiming_User_System(&read_time, &user_start, &user_end, &system_start, &system_end);
	
	// Write read time.
	writeReadTime(read_time);
	
	// Processing time start.
    #ifdef FORCE_SINGLE_CORE
    	start_timing(&usage, &user_start, &system_start);
    #else
    	gettimeofday(&time_start, NULL);
    #endif
	
	Thread thread[g_maxThreads];
	
    if (size <= g_truncateSize) 		 // Small enough to run single core
       single_core = 1;
    
    #ifdef FORCE_SINGLE_CORE
    	single_core = 1;
    #endif
	#ifdef FORCE_MULTI_CORE		 // If max number of cores is equal to 1, will run single core
    	single_core = 0;
    #endif
    	
    if (numCores >= 2 && !single_core) { // If multiple cores, matrix size not small and multi core not forced
        single_core = 0;
        numThreads = min(numCores, 7); 	 // Using at most 7 threads
    } else {
        single_core = 1;
    }
    
    if ( size <= 2 ) {
    	handleCornerCase(matrix_A, matrix_B, matrix_C, size); // Corner cases, n = 0 or n = 1 
	} else {
		if (single_core) {  
		    Array2D<E> matrix_resultA(size/2, size/2, 1);   // Buffer for addition, subtraction and copy results
		    Array2D<E> matrix_resultB(size/2, size/2, 1);   // Buffer for addition, subtraction and copy results
			strassenMultiplication(matrix_A, matrix_B, 0, 0, size, matrix_C, matrix_resultA, matrix_resultB);	
		} else {
		    parallelStrassenMultiplication(matrix_A, matrix_B, numThreads, size, &matrix_C, thread);
		}
    }
    
    
    // Processing time end.
	#ifdef FORCE_SINGLE_CORE
    	end_timing(&usage, &user_end, &system_end);
    	getTiming_User_System(&process_time, &user_start, &user_end, &system_start, &system_end);
    #else
    	gettimeofday(&time_end, NULL);
    	getTiming(&process_time, &time_start, &time_end);
    #endif
	
	// Write processing time.
	writeProcessTime(process_time);
	
	// Timing start.
	start_timing(&usage, &user_start, &system_start);
    
    writeOutput(matrix_C, size, size);
    
    // Timing end.
	end_timing(&usage, &user_end, &system_end);
	
	// Get Time.
	getTiming_User_System(&write_time, &user_start, &user_end, &system_start, &system_end);
	
	// Write writing time.
	writeWriteTime(write_time);

    return 0;
}



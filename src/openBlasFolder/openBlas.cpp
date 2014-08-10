#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include <iomanip> 	
#include <thread>
#include <cblas.h>
#include "Array2D.h"

typedef double E;

using namespace std;

const int g_outputWidth = 9; 	   // Set the output width for elements.
const int g_outputPrecision = 15;  // Matrix C elements output precision.

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
					file_matrix_B >> matrix_B(i,j); 	
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
		file_out.open("openBlas_matrix_C.txt");
	#else
		file_out.open("openBlasParallel_matrix_C.txt");
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
		file_out.open("openBlas_read_time.txt");
	#else
		file_out.open("openBlasParallel_read_time.txt");
	#endif

	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeWriteTime(double time){
	ofstream file_out;
	#ifdef FORCE_SINGLE_CORE
		file_out.open("openBlas_write_time.txt");
	#else
		file_out.open("openBlasParallel_write_time.txt");
	#endif
	
	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeProcessTime(double time){
	ofstream file_out;
	#ifdef FORCE_SINGLE_CORE
		file_out.open("openBlas_process_time.txt");
	#else
		file_out.open("openBlasParallel_process_time.txt");
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

    double read_time, process_time, write_time;
	struct timeval system_start, system_end, user_start, user_end; // Structs used by rusage
	struct rusage usage; 
	struct timeval time_start, time_end; // Structs gettimeofday
	
	// Timing start.
	start_timing(&usage, &user_start, &system_start);
		
	// Get matrices dimension.
	int size = n;

	// Create
	Array2D <E> matrix_A(m, k);
	Array2D <E> matrix_B(k, n);
	Array2D <E> matrix_C(m, n, 0, 0); //Fill Zeros

    // Read Matrices A and B from input
    if (readInput(matrix_A, matrix_B, m, k, n) != 1)  // If return is not 1, something is wrong
		return 0;
	
	// Timing end.
	end_timing(&usage, &user_end, &system_end);
	
	// Get Time.
	getTiming_User_System(&read_time, &user_start, &user_end, &system_start, &system_end);
	
	// Write read time.
	writeReadTime(read_time);
	
	#ifdef FORCE_SINGLE_CORE
		goto_set_num_threads(0);
	#else
		goto_set_num_threads(std::thread::hardware_concurrency());
	#endif
	
	// Processing time start.
    #ifdef FORCE_SINGLE_CORE
    	start_timing(&usage, &user_start, &system_start);
    #else
    	gettimeofday(&time_start, NULL);
    #endif
    	
    cblas_dgemm ( CblasRowMajor, CblasNoTrans, CblasNoTrans , m , n , k , 1.0 , matrix_A.getAddress(0) , k , matrix_B.getAddress(0) , 					  n , 0.0 , matrix_C.getAddress(0) , n );
    
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
	   
	writeOutput(matrix_C, m, n);
	
	// Timing end.
	end_timing(&usage, &user_end, &system_end);
	
	// Get Time.
	getTiming_User_System(&write_time, &user_start, &user_end, &system_start, &system_end);
	
	// Write writing time.
	writeWriteTime(write_time);

	
    return 0;
}


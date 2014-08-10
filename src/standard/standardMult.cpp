#include <iostream>
#include <fstream> 
#include <sys/time.h>
#include <sys/resource.h>
#include <iomanip> // Output formatting

using namespace std;

const int g_outputWidth = 9; 		// Set the output width for elements.
const int g_outputPrecision = 15;	// Output elements precision.

template<class E> class init
{

public:
	void matrixMultiplication(int A_dim1, int A_dim2, int B_dim1, E** A, E** B, E** C){
		int i, j, k, l;
		for (i = 0; i < A_dim1; i++) {
			for (j = 0; j <B_dim1; j++) {
		    	for (k = 0; k < A_dim2; k++) {
		        	C[i][j] += A[i][k]* B[k][j];
		      	}
		  	}
	   	}	  
	   	return;
	}

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
						file_matrix_B >> matrix_B[i][j]; 	
				}
			}
		} else return 0;
	
		file_matrix_B.close();

		return 1;
	}

	// Outputs the resulting matrix C. C = A*B
	void writeOutput(E** matrix_C, int dim1, int dim2) {
		ofstream file_out;
		file_out.open("standardMult_matrix_C.txt");
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

	void getTiming(double* timing, struct timeval* time_start, struct timeval* time_end) {
		*timing = 0;
	
		*timing = (double)(time_end->tv_sec- time_start->tv_sec)*1000;
		*timing += (double)(time_end->tv_usec - time_start->tv_usec)/1000;
	
		return;				   
	}

	void writeReadTime(double time){
		ofstream file_out;
		file_out.open("standardMult_read_time.txt");
		if (file_out.is_open()) {
			file_out << time;
			file_out.close();
		}	
	}

	void writeWriteTime(double time){
		ofstream file_out;
		file_out.open("standardMult_write_time.txt");
		if (file_out.is_open()) {
			file_out << time;
			file_out.close();
		}	
	}

	void writeProcessTime(double time){
		ofstream file_out;
		file_out.open("standardMult_process_time.txt");
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
	
	init(int m, int k, int n) {
		
		double read_time, process_time, write_time;
		double aux;
		struct timeval system_start, system_end, user_start, user_end; // Structs used by rusage
		struct rusage usage;
	
		// Timing start.
		start_timing(&usage, &user_start, &system_start);
	
		E** matrix_A;
		E** matrix_B;
		E** matrix_C;
	
		int i, j;
	
		matrix_A = (E**) malloc (m * sizeof(E*) );
		matrix_B = (E**) malloc (k * sizeof(E*) );
		matrix_C = (E**) malloc (m * sizeof(E*) );

		// Create matrix A
		for (i = 0; i < m; ++i)
			matrix_A[i] = (E*) malloc (k * sizeof(E));
		
		// Create matrix B
		for (i = 0; i < k; ++i)
			matrix_B[i] = (E*) malloc (n * sizeof(E));
	
		// Fill matrix C with 0
		for (i = 0; i < m; ++i) {
			matrix_C[i] = (E*) malloc (n * sizeof(E));
			for (j = 0; j < n; ++j)
				matrix_C[i][j] = 0;
		}
	
		// Read Matrices A and B from input
		if (readInput(matrix_A, matrix_B, m, k, n) != 1)  // If return is not 1, something is wrong
			return;
	
		// Timing end.
		end_timing(&usage, &user_end, &system_end);
	
		// Get Time.
		getTiming_User_System(&read_time, &user_start, &user_end, &system_start, &system_end);
	
		// Write read time.
		writeReadTime(read_time);
	
		// Timing start.
		start_timing(&usage, &user_start, &system_start);
	
		matrixMultiplication(m, k, n, matrix_A, matrix_B, matrix_C);	
	
		// Timing end.
		end_timing(&usage, &user_end, &system_end);
	
		// Get Time.
		getTiming_User_System(&process_time, &user_start, &user_end, &system_start, &system_end);
	
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

int main(int argc, char *argv[]) {	
	int m, k, n, cacheBlockSize, dataType;
	
	if (!readDimensions(&m, &k, &n, &cacheBlockSize, &dataType)) { 
		cout << "Couldn't read in.txt file!" << endl;
		return 0;	
	}	

	if (dataType == 0) {
		init<int>(m, k, n);
	} else if (dataType == 1) {
		init<float>(m, k, n);
	} else {
		init<double>(m, k, n);
	}
			
    return 0;
}



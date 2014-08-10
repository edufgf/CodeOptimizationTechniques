/* clMath Version 1.10 */

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <iomanip> 
#include "Array2D.h"

/* Include CLBLAS header. It automatically includes needed OpenCL header,
 * so we can drop out explicit inclusion of cl.h header.
 */
#include <clAmdBlas.h>

/* This example uses predefined matrices and their characteristics for
 * simplicity purpose.
 */

using namespace std;

typedef cl_float E; // Define the type of elements being used.

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
	file_out.open("clMath_matrix_C.txt");
		
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
	file_out.open("clMath_read_time.txt");

	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeWriteTime(double time){
	ofstream file_out;
	file_out.open("clMath_write_time.txt");
	
	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeProcessTime(double time){
	ofstream file_out;
	file_out.open("clMath_process_time.txt");

	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeProcessTimeReal(double time){
	ofstream file_out;
	file_out.open("clMath_process_time_real.txt");

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
    double process_time_real;
	struct timeval system_start, system_end, user_start, user_end; // Structs used by rusage
	struct rusage usage; 
	struct timeval time_start, time_end; // Structs gettimeofday
	struct timeval time_start2, time_end2;
	
	// OpenCL vars
	cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;
    int ret = 0;
	
	
	// Timing start.
	start_timing(&usage, &user_start, &system_start);
	
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
	
	// Processing time start.
	gettimeofday(&time_start, NULL);
	
	// Setup OpenCL environment.
    err = clGetPlatformIDs(1, &platform, NULL);

    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetDeviceIDs() failed with %d\n", err );
        return 1;
    }

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return 1;
    }

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }
    
    // Setup clAmdBlas.
    err = clAmdBlasSetup();
    if (err != CL_SUCCESS) {
        printf("clAmdBlasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }
	
	// Prepare OpenCL memory objects and place matrices inside them.
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, m * k * sizeof(matrix_A(0)),
                          NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, k * n * sizeof(matrix_B(0)),
                          NULL, &err);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m * n * sizeof(matrix_C(0)),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
        m * k * sizeof(matrix_A(0)), matrix_A.getAddress(0), 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
        k * n * sizeof(matrix_B(0)), matrix_B.getAddress(0), 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
        m * n * sizeof(matrix_C(0)), matrix_C.getAddress(0), 0, NULL, NULL);
	
	// Processing real time start.
	gettimeofday(&time_start2, NULL);
	
    // Call clAmdBlas function.
    err = clAmdBlasSgemm(clAmdBlasRowMajor, clAmdBlasNoTrans, clAmdBlasNoTrans, m, n, k, 1, bufA,
                         k, bufB, n, 1, bufC, n, 1, &queue,
                         0, NULL, &event);
                         
    if (err != CL_SUCCESS) {
        printf("clAmdBlasSgemm() failed with %d\n", err);
        ret = 1;
    }
    else {
        // Wait for calculations to be finished.
        err = clWaitForEvents(1, &event);
		
		// Processing time end.
		gettimeofday(&time_end2, NULL);
    	getTiming(&process_time_real, &time_start2, &time_end2);
	
		// Write processing time.
		writeProcessTimeReal(process_time_real);
		
        // Fetch results of calculations from GPU memory.
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                                  m * n * sizeof(matrix_C(0)),
                                  matrix_C.getAddress(0), 0, NULL, NULL);
    }
    

    // Release OpenCL memory objects.
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);

    // Finalize work with clAmdBlas. 
    clAmdBlasTeardown();

    // Release OpenCL working objects.
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
	
	// Processing time end.
	gettimeofday(&time_end, NULL);
    getTiming(&process_time, &time_start, &time_end);
	
	// Write processing time.
	writeProcessTime(process_time);
	
	// Timing start.
	gettimeofday(&time_start, NULL);
    
    writeOutput(matrix_C, m, n);
    
    // Timing end.
    gettimeofday(&time_end, NULL);
	
	// Get Time.
	getTiming(&write_time, &time_start, &time_end);
	
	// Write writing time.
	writeWriteTime(write_time);

    return 0;
}

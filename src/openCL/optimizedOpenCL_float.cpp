#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include <iomanip> 
#include <CL/cl.h>
#include "Array2D.h"

using namespace std;

typedef float E; // Define the type of elements being used.

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
	file_out.open("optimizedOpenCL_matrix_C.txt");
		
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
	file_out.open("optimizedOpenCL_read_time.txt");
	
	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeWriteTime(double time){
	ofstream file_out;
	file_out.open("optimizedOpenCL_write_time.txt");

	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeProcessTime(double time){
	ofstream file_out;
	file_out.open("optimizedOpenCL_process_time.txt");

	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeProcessTimeReal(double time){
	ofstream file_out;
	file_out.open("optimizedOpenCL_process_time_real.txt");

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
	
	
	// Timing start.
	start_timing(&usage, &user_start, &system_start);
	
    // Create
	Array2D <E> matrix_A(m, k);
	Array2D <E> matrix_B(k, n);
	Array2D <E> matrix_C(m, n); //Receive output by GPU
	
	// Read Matrices A and B from input
    if (readInput(matrix_A, matrix_B, m, k, n) == 0) {
    	cout << "Couldn't open files to read matrices!" << endl;
    	return 0;
    }

    // Timing end.
	end_timing(&usage, &user_end, &system_end);
	
	// Get Time.
	getTiming_User_System(&read_time, &user_start, &user_end, &system_start, &system_end);
	
	// Write read time.
	writeReadTime(read_time);

	// Processing time start.
	gettimeofday(&time_start, NULL);

	int err;
    size_t global;                  // global domain size  

    cl_device_id     device_id;     // compute device id 
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        ko_vadd;       // compute kernel
    
    cl_mem d_a;                     // device memory used for the input  a vector
    cl_mem d_b;                     // device memory used for the input  b vector
    cl_mem d_c;                     // device memory used for the output c vector
    
    // Set up platform and GPU device
    cl_uint numPlatforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms <= 0) {
        printf("Error: Failed to find a platform!\n%d\n",err);
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    if (err != CL_SUCCESS || numPlatforms <= 0) {
        printf("Error: Failed to get the platform!\n%d\n",err);
        return EXIT_FAILURE;
    }

    // Secure a GPU
    for (int i = 0; i < numPlatforms; i++) {
        err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
            break;
    }

    if (device_id == NULL) {
        printf("Error: Failed to create a device group!\n%d\n",err);
        return EXIT_FAILURE;
    }

    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n%d\n", err);
        return EXIT_FAILURE;
    }
	
    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n%d\n", err);
        return EXIT_FAILURE;
    }
	
	// Read kernel file
	string kernel_str;
	ifstream file_input("openCL/optimizedOpenCL_float.cl");
	stringstream buffer;
	buffer << file_input.rdbuf();
	kernel_str = buffer.str();

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_str, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n%d\n", err);
        return EXIT_FAILURE;
    }

    // Build the program  
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n%d\n", err);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program 
    ko_vadd = clCreateKernel(program, "mat_mult", &err);
    if (!ko_vadd || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n%d\n", err);
        return EXIT_FAILURE;
    }

    // Create the input (a, b) and output (c) arrays in device memory
    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(E) * m * k, NULL, NULL);
    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(E) * k * n, NULL, NULL);
    d_c  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(E) * m * n, NULL, NULL);
    
    // Clean Buffer 
    // double zero = 0.0;
	// clEnqueueFillBuffer ( commands , d_c, &zero, sizeof(double), 0, sizeof(double) * M*N, 0 , NULL,NULL);
	
    if (!d_a || !d_b || !d_c) {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
    // Write a and b vectors into compute device memory 
    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(E) * m * k, matrix_A.getAddress(0), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write h_a to source array!\n%d\n", err);
        exit(1);
    }

    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(E) * k * n, matrix_B.getAddress(0), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write h_b to source array!\n%d\n", err);
        exit(1);
    }
	
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(ko_vadd, 3, sizeof(int), &m);
    err |= clSetKernelArg(ko_vadd, 4, sizeof(int), &n);
    err |= clSetKernelArg(ko_vadd, 5, sizeof(int), &k);
    
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments!\n");
        exit(1);
    }
	
	// Global & Local work size
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = 16;
	localWorkSize[1] = 16;
	globalWorkSize[0] = m;
	globalWorkSize[1] = n;
	
	// Processing time real start.
	gettimeofday(&time_start2, NULL);
 	
    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n%d\n", err);
        return EXIT_FAILURE;
    }
	
    // Wait for the commands to complete before stopping the timer
    clFinish(commands);

	// Processing real time end.
	gettimeofday(&time_end2, NULL);
	getTiming(&process_time_real, &time_start2, &time_end2);
	
	// Write processing real time
	writeProcessTimeReal(process_time_real);

    // Read back the results from the compute device
    err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(E) * m * n, matrix_C.getAddress(0), 0, NULL, NULL );  
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array!\n%d\n", err);
        exit(1);
    }
	
    // Cleanup then shutdown
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);	
	    
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

/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <iomanip> 
#include "AMDopenCL_double.hpp"

using namespace std;

const int g_outputWidth = 9; 	   // Set the output width for elements.
const int g_outputPrecision = 15;  // Matrix C elements output precision.


int MatrixMultiplication::readInput(int m, int k, int n) {
	ifstream file_matrix_A;
	ifstream file_matrix_B;
	file_matrix_A.open ("matrix_A.txt", std::ifstream::in); //File to read matrix A
	file_matrix_B.open ("matrix_B.txt", std::ifstream::in); //File to read matrix B
	
	int dim1 = m;
	int dim2 = k;

	if (file_matrix_A.is_open()) {
		for (int i = 0; i < dim1; ++i) {
			for (int j = 0; j < dim2; ++j) {
					file_matrix_A >> input0[i*dim1+j]; 	
			}
		}
	} else return 0;
	
	dim1 = k;
	dim2 = n;
	
	if (file_matrix_B.is_open()) {
		for (int i = 0; i < dim1; ++i) {
			for (int j = 0; j < dim2; ++j) {
					file_matrix_B >> input1[i*dim1+j]; 	
			}
		}
	} else return 0;
	
	file_matrix_B.close();

	return 1;
}

// Outputs the resulting matrix C. C = A*B
void MatrixMultiplication::writeOutput() {
	ofstream file_out;
	file_out.open("AMDopenCL_matrix_C.txt");
	int dim1 = m;
	int dim2 = n;
	if (file_out.is_open()) {
		file_out << std::fixed;
		for (int i = 0; i < dim1; ++i) {
			for (int j = 0; j < dim2; ++j) {
				file_out << std::setprecision(g_outputPrecision) << std::setw(g_outputWidth) << output[i*dim1+j] << " ";
		  	}
		  	file_out << endl;
		}	
	} 
}

int MatrixMultiplication::readDimensions(int* m, int* k, int* n, int* cacheBlockSize, int* dataType) {
	ifstream file_input;
	file_input.open ("matrix_dim.txt", std::ifstream::in); //File to read matrices dimensions
	if (file_input.is_open()) {
		file_input >> *m >> *k >> *n >> *cacheBlockSize >> *dataType;
		file_input.close();
	} else return 0;
	return 1;
}

void getTiming(double* timing, struct timeval* time_start, struct timeval* time_end) {
	*timing = 0;
	
	*timing = (double)(time_end->tv_sec- time_start->tv_sec)*1000;
	*timing += (double)(time_end->tv_usec - time_start->tv_usec)/1000;
	
	return;				   
}

void writeReadTime(double time){
	ofstream file_out;
	file_out.open("AMDopenCL_read_time.txt");

	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeWriteTime(double time){
	ofstream file_out;
	file_out.open("AMDopenCL_write_time.txt");
	
	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeProcessTime(double time){
	ofstream file_out;
	file_out.open("AMDopenCL_process_time.txt");

	if (file_out.is_open()) {
		file_out << time;
		file_out.close();
	}	
}

void writeProcessTimeReal(double time){
	ofstream file_out;
	file_out.open("AMDopenCL_process_time_real.txt");

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


int
MatrixMultiplication::setupMatrixMultiplication()
{
    // allocate and init memory used by host  input0[width0][height0]
    cl_uint inputSizeBytes0 = width0 * height0 * sizeof(cl_double);

    input0 = (cl_double *) malloc(inputSizeBytes0);
    CHECK_ALLOCATION(input0, "Failed to allocate host memory. (input0)");

    // allocate and init memory used by host input1[width1][height1]
    cl_uint inputSizeBytes1 = width1 * height1 * sizeof(cl_double);

    input1 = (cl_double *) malloc(inputSizeBytes1);
    CHECK_ALLOCATION(input1, "Failed to allocate host memory. (input1)");

    // Read input
    readInput(m, k, n); 

    // allocate memory for output[width1][height0]
    cl_uint outputSizeBytes = height0 * width1 * sizeof(cl_double);

    output = (cl_double *) malloc(outputSizeBytes);
    CHECK_ALLOCATION(output, "Failed to allocate host memory. (output)");

    return SDK_SUCCESS;
}

/**
 * genBinary Image function is used to when we want to create the binary
 * for the kernel and run it at a later time. This is useful where offline
 * compilation is the preferred mechanism.
 */
int
MatrixMultiplication::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("openCL/MatrixMultiplication_Kernels_double.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}



int
MatrixMultiplication::setupCL(void)
{
    cl_int status = 0;
    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    //retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");

    // creating context
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = clCreateContextFromType(
                  cps,
                  dType,
                  NULL,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR(status, "clCreateContextFromType failed.");

    // getting device on which to run the sample
    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        if(!eAppGFLOPS)
        {
            prop |= CL_QUEUE_PROFILING_ENABLE;
        }

        commandQueue = clCreateCommandQueue(
                           context,
                           devices[sampleArgs->deviceId],
                           prop,
                           &status);
        CHECK_OPENCL_ERROR(status, "clCreateCommandQueue failed.");
    }

    // Set Presistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
    if(sampleArgs->isAmdPlatform())
    {
        inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
    }

    // Create buffer for matrix A
    inputBuffer0 = clCreateBuffer(
                       context,
                       inMemFlags,
                       sizeof(cl_double) * width0 * height0,
                       0,
                       &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (inputBuffer0)");

    // Create buffer for matrix B
    inputBuffer1 = clCreateBuffer(
                       context,
                       inMemFlags,
                       sizeof(cl_double) * width1 * height1,
                       0,
                       &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (inputBuffer1)");

    outputBuffer = clCreateBuffer(
                       context,
                       CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                       sizeof(cl_double) * height0 * width1,
                       0,
                       &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (outputBuffer)");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("openCL/MatrixMultiplication_Kernels_double.cl");
    buildData.devices = devices;
    buildData.deviceId = sampleArgs->deviceId;
    buildData.flagsStr = std::string("");
    if(sampleArgs->isLoadBinaryEnabled())
    {
        buildData.binaryName = std::string(sampleArgs->loadBinary.c_str());
    }

    if(sampleArgs->isComplierFlagsSpecified())
    {
        buildData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    retValue = buildOpenCLProgram(program, context, buildData);
    CHECK_ERROR(retValue, SDK_SUCCESS, "buildOpenCLProgram() failed");

    // If local memory is present then use the specific kernel
    if(lds)
    {
        kernel = clCreateKernel(program, "mmmKernel_local", &status);
    }
    else
    {
        kernel = clCreateKernel(program, "mmmKernel", &status);
    }
    CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");
    return SDK_SUCCESS;
}

int
MatrixMultiplication::setWorkGroupSize()
{
    /*
     * Kernel runs over complete output matrix with blocks of blockSize x blockSize
     * running concurrently
     */
    cl_int status = 0;
    globalThreads[0] = width1 / 4;
    globalThreads[1] = height0/ 4;
    localThreads[0] = blockSize;
    localThreads[1] = blockSize;

    // Setting the KernelWorkGroupInfo values
    status = kernelInfo.setKernelWorkGroupInfo(kernel,
             devices[sampleArgs->deviceId]);
    CHECK_ERROR(status,0, "setKernelWrkGroupInfo failed");

    availableLocalMemory = deviceInfo.localMemSize - kernelInfo.localMemoryUsed;
    neededLocalMemory    = 2 * blockSize * blockSize * sizeof(cl_double);
    if(neededLocalMemory > availableLocalMemory)
    {
        std::cout << "Unsupported: Insufficient local memory on device." << std::endl;
        return SDK_SUCCESS;
    }

    if((cl_uint)(localThreads[0] * localThreads[1]) >
            kernelInfo.kernelWorkGroupSize)
    {
        if(kernelInfo.kernelWorkGroupSize >= 64)
        {
            blockSize = 8;
            localThreads[0] = blockSize;
            localThreads[1] = blockSize;
        }
        else if(kernelInfo.kernelWorkGroupSize >= 32)
        {
            blockSize = 4;
            localThreads[0] = blockSize;
            localThreads[1] = blockSize;
        }
        else
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << localThreads[0] * localThreads[1] <<
                      std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfo.kernelWorkGroupSize<<std::endl;
            return SDK_FAILURE;
        }
    }

    if(localThreads[0] > deviceInfo.maxWorkItemSizes[0] ||
            localThreads[1] > deviceInfo.maxWorkItemSizes[1] ||
            localThreads[0] * localThreads[1] > deviceInfo.maxWorkGroupSize)
    {
        std::cout <<
                  "Unsupported: Device does not support requested number of work items." <<
                  std::endl;
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}


int MatrixMultiplication::runCLKernels(void)
{
    cl_int   status;
    status = setWorkGroupSize();
    CHECK_ERROR(status, SDK_SUCCESS, "getWorkGroupSize() failed");

    cl_event ndrEvt;
    cl_int eventStatus = CL_QUEUED;

    // Set input data to matrix A and matrix B
    cl_event inMapEvt1, inMapEvt2, inUnmapEvt1, inUnmapEvt2, outMapEvt, outUnmapEvt;
    void* mapPtr1 = clEnqueueMapBuffer(
                        commandQueue,
                        inputBuffer0,
                        CL_FALSE,
                        CL_MAP_WRITE,
                        0,
                        width0 * height0 * sizeof(cl_double),
                        0,
                        NULL,
                        &inMapEvt1,
                        &status);
    CHECK_OPENCL_ERROR(status, "clEnqueueMapBuffer failed. (inputBuffer0)");

    void* mapPtr2 = clEnqueueMapBuffer(
                        commandQueue,
                        inputBuffer1,
                        CL_FALSE,
                        CL_MAP_WRITE,
                        0,
                        width1 * height1 * sizeof(cl_double),
                        0,
                        NULL,
                        &inMapEvt2,
                        &status);
    CHECK_OPENCL_ERROR(status, "clEnqueueMapBuffer failed. (inputBuffer1)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&inMapEvt1);
    CHECK_ERROR(status,SDK_SUCCESS, "WaitForEventAndRelease(inMapEvt1) Failed");
    memcpy(mapPtr1, input0, sizeof(cl_double) * width0  * height0);

    status = waitForEventAndRelease(&inMapEvt2);
    CHECK_ERROR(status,SDK_SUCCESS, "WaitForEventAndRelease(inMapEvt2) Failed");
    memcpy(mapPtr2, input1, sizeof(cl_double) * width1  * height1);

    status = clEnqueueUnmapMemObject(
                 commandQueue,
                 inputBuffer0,
                 mapPtr1,
                 0,
                 NULL,
                 &inUnmapEvt1);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed. (inputBuffer0)");

    status = clEnqueueUnmapMemObject(
                 commandQueue,
                 inputBuffer1,
                 mapPtr2,
                 0,
                 NULL,
                 &inUnmapEvt2);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed. (inputBuffer1)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&inUnmapEvt1);
    CHECK_ERROR(status, SDK_SUCCESS, "waitForEventAndRelease(inUnmapEvt1) failed");

    status = waitForEventAndRelease(&inUnmapEvt2);
    CHECK_ERROR(status,SDK_SUCCESS, "waitForEventAndRelease(inUnmapEvt2) failed");

    // Set appropriate arguments to the kernel

    // output array as the 1st argument : stores product of input0 and input1
    status = clSetKernelArg(
                 kernel,
                 0,
                 sizeof(cl_mem),
                 (void *)&inputBuffer0);
    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (inputBuffer0)");

    // the input matrix  as 2nd argument - input0
    status = clSetKernelArg(
                 kernel,
                 1,
                 sizeof(cl_mem),
                 (void *)&inputBuffer1);
    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (inputBuffer1)");

    // the input matrix as 3rd argument - input1
    status = clSetKernelArg(
                 kernel,
                 2,
                 sizeof(cl_mem),
                 (void *)&outputBuffer);
    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (outputBuffer)");

    // width0 of the input0 matrix as 4th argument - width0
    status = clSetKernelArg(
                 kernel,
                 3,
                 sizeof(cl_int),
                 (void*)&width0);
    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (width0)");

    // Set local memory argument if Scratchpad is available
    if(lds)
    {
        status = clSetKernelArg(
                     kernel,
                     4,
                     (blockSize * 4) * (blockSize * 4) * sizeof(cl_double),
                     NULL);
        CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory)");
    }
    else
    {
        status = clSetKernelArg(kernel, 4, sizeof(cl_int), &width1);
        CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (width1)");
    }
	
	double process_time_real;
	struct timeval time_start2, time_end2;
	
	// Processing time real start.
	gettimeofday(&time_start2, NULL);
	
    // Enqueue a kernel run call
    status = clEnqueueNDRangeKernel(
                 commandQueue,
                 kernel,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 &ndrEvt);
    CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");
	
    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    // wait for the kernel call to finish execution
    eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = clGetEventInfo(
                     ndrEvt,
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     sizeof(cl_int),
                     &eventStatus,
                     NULL);
        CHECK_OPENCL_ERROR(status, "clGetEventInfo failed.");
    }


	// Processing real time end.
	gettimeofday(&time_end2, NULL);
	getTiming(&process_time_real, &time_start2, &time_end2);
	
	// Write processing real time
	writeProcessTimeReal(process_time_real);
	
    if(!eAppGFLOPS)
    {
        // Calculate performance
        cl_ulong startTime;
        cl_ulong endTime;

        // Get kernel profiling info
        status = clGetEventProfilingInfo(ndrEvt,
                                         CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong),
                                         &startTime,
                                         0);
        CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.(startTime)");

        status = clGetEventProfilingInfo(ndrEvt,
                                         CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong),
                                         &endTime,
                                         0);
        CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.(endTime)");

        // Print performance numbers
        double sec = 1e-9 * (endTime - startTime);
        kernelTime += sec;
    }

    status = clReleaseEvent(ndrEvt);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent failed. (ndrEvt)");

    void* outMapPtr = clEnqueueMapBuffer(
                          commandQueue,
                          outputBuffer,
                          CL_FALSE,
                          CL_MAP_READ,
                          0,
                          width1 * height0 * sizeof(cl_double),
                          0,
                          NULL,
                          &outMapEvt,
                          &status);
    CHECK_OPENCL_ERROR(status, "clEnqueueMapBuffer failed. (outputBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&outMapEvt);
    CHECK_ERROR(status,0, "waitForEventAndRelease(outMapEvt) failed");
    memcpy(output, outMapPtr, sizeof(cl_double) * width1  * height0);

    status = clEnqueueUnmapMemObject(
                 commandQueue,
                 outputBuffer,
                 outMapPtr,
                 0,
                 NULL,
                 &outUnmapEvt);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed. (outputBuffer)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&outUnmapEvt);
    CHECK_ERROR(status,0, "waitForEventAndRelease(outUnmapEvt) failed");

    return SDK_SUCCESS;
}


/*
 * This is a naive O(N^3) CPU implementation of matrix multiplication
 */
void
MatrixMultiplication::matrixMultiplicationCPUReference(
    cl_double * output,
    cl_double * input0,
    cl_double * input1,
    const cl_uint y,
    const cl_uint x,
    const cl_uint z)
{
    for(cl_uint i = 0; i < y; i++)
    {
        for(cl_uint j = 0; j < z; j++)
        {
            for(cl_uint k = 0; k < x; k++)
            {
                output[i * z + j] += (input0[i * x + k] * input1[k * z + j]);
            }
        }
    }
}

int
MatrixMultiplication::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // add an option for getting blockSize from commandline
    Option* xParam = new Option;
    CHECK_ALLOCATION(xParam, "Memory Allocation error.\n");
    xParam->_sVersion = "x";
    xParam->_lVersion = "height0";
    xParam->_description = "height of matrix A";
    xParam->_type     = CA_ARG_INT;
    xParam->_value    = &n;
    sampleArgs->AddOption(xParam);
    delete xParam;

    Option* yParam = new Option;
    CHECK_ALLOCATION(yParam, "Memory Allocation error.\n");
    yParam->_sVersion = "y";
    yParam->_lVersion = "width0";
    yParam->_description = "width of matrix A and Height of matrix B";
    yParam->_type     = CA_ARG_INT;
    yParam->_value    = &m;
    sampleArgs->AddOption(yParam);
    delete yParam;

    Option* zParam = new Option;
    CHECK_ALLOCATION(zParam, "Memory Allocation error.\n");
    zParam->_sVersion = "z";
    zParam->_lVersion = "width1";
    zParam->_description = "width of matrix B";
    zParam->_type     = CA_ARG_INT;
    zParam->_value    = &k;
    sampleArgs->AddOption(zParam);
    delete zParam;

    Option* blockSizeParam = new Option;
    CHECK_ALLOCATION(blockSizeParam, "Memory Allocation error.\n");
    blockSizeParam->_sVersion = "b";
    blockSizeParam->_lVersion = "blockSize";
    blockSizeParam->_description =
        "Use local memory of dimensions blockSize x blockSize";
    blockSizeParam->_type     = CA_ARG_INT;
    blockSizeParam->_value    = &blockSize;
    sampleArgs->AddOption(blockSizeParam);
    delete blockSizeParam;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory Allocation error.\n");
    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;
    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

    Option* appGflops_option = new Option;
    CHECK_ALLOCATION(appGflops_option, "Memory Allocation error.\n");
    appGflops_option->_sVersion = "";
    appGflops_option->_lVersion = "eAppGflops";
    appGflops_option->_description =
        "Prints GFLOPS calculated from transfer + kernel time";
    appGflops_option->_type = CA_NO_ARGUMENT;
    appGflops_option->_value = &eAppGFLOPS;
    sampleArgs->AddOption(appGflops_option);
    delete appGflops_option;

    return SDK_SUCCESS;
}

int
MatrixMultiplication::setup()
{
 
 	int cacheBlockSize, dataType;
	
	if (!readDimensions(&m, &k, &n, &cacheBlockSize, &dataType)) { 
		cout << "Couldn't read in.txt file!" << endl;
		return 0;	
	}	

    // Make sure the dimensions are multiples of blockSize
    const int vectorSize = 4;
    if(n % (blockSize * vectorSize) != 0)
    {
        n = (n / (blockSize * vectorSize) + 1) * (blockSize * vectorSize);
    }

    if(m % (blockSize * vectorSize) != 0)
    {
        m = (m / (blockSize * vectorSize) + 1) * (blockSize * vectorSize);
    }

    if(k % (blockSize * vectorSize) != 0)
    {
        k = (k / (blockSize * vectorSize) + 1) * (blockSize * vectorSize);
    }

    width0  = k;
    height0 = m;

    width1  = n;
    height1 = k;

    if(setupMatrixMultiplication()!=SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}


int MatrixMultiplication::run() {
    int kernelRun = runCLKernels();
    if(kernelRun != SDK_SUCCESS)
    {
        return kernelRun;
    }
    return SDK_SUCCESS;
}

int
MatrixMultiplication::verifyResults()
{
    if(sampleArgs->verify)
    {
        // reference implementation
        matrixMultiplicationCPUReference(verificationOutput, input0, input1, height0,
                                         width0,  width1);

        // compare the results and see if they match
        if(compare(output, verificationOutput, height0*width1))
        {
            std::cout<<"Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout<<"Failed\n" << std::endl;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void
MatrixMultiplication::printStats()
{
    if(sampleArgs->timing)
    {
        if(eAppGFLOPS)
        {
            std::string strArray[4] = {"MatrixA", "MatrixB", "Time(sec)", "[Transfer+kernel]Time(sec)"};
            std::string stats[4];

            double flops = 2 * width0 * width1;
            double perf = (flops / appTime) * height0 * 1e-9;
            if(sampleArgs->timing)
            {
                std::cout << "GFlops achieved : " << perf << std::endl << std::endl;
            }

            sampleTimer->totalTime = setupTime + appTime;

            stats[0]  = toString(height0, std::dec)
                        +"x"+toString(width0, std::dec);
            stats[1]  = toString(height1, std::dec)
                        +"x"+toString(width1, std::dec);
            stats[2]  = toString(sampleTimer->totalTime, std::dec);
            stats[3]  = toString(appTime, std::dec);

            printStatistics(strArray, stats, 4);
        }
        else
        {
            std::string strArray[4] = {"MatrixA", "MatrixB", "Time(sec)", "kernelTime(sec)"};
            std::string stats[4];

            double flops = 2 * width0 * width1;
            double perf = (flops / kernelTime) * height0 * 1e-9;
            if(sampleArgs->timing)
            {
                std::cout << "GFlops achieved : " << perf << std::endl << std::endl;
            }

            sampleTimer->totalTime = setupTime + kernelTime;

            stats[0]  = toString(height0, std::dec)
                        +"x"+toString(width0, std::dec);
            stats[1]  = toString(height1, std::dec)
                        +"x"+toString(width1, std::dec);
            stats[2]  = toString(sampleTimer->totalTime, std::dec);
            stats[3]  = toString(kernelTime, std::dec);

            printStatistics(strArray, stats, 4);
        }
    }
}

int
MatrixMultiplication::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseMemObject(inputBuffer0);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer0)");

    status = clReleaseMemObject(inputBuffer1);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(inputBuffer1)");

    status = clReleaseMemObject(outputBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(outputBuffer)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    // release program resources (input memory etc.)

    //FREE(input0);
    //FREE(input1);
    //FREE(output);
    FREE(verificationOutput);
    FREE(devices);

    return SDK_SUCCESS;
}


int main(int argc, char * argv[]) {
	double read_time, process_time, write_time;
    double process_time_real;
	struct timeval system_start, system_end, user_start, user_end; // Structs used by rusage
	struct rusage usage; 
	struct timeval time_start, time_end; // Structs gettimeofday
	struct timeval time_start2, time_end2;

    MatrixMultiplication clMatrixMultiplication;

	// Timing start
	start_timing(&usage, &user_start, &system_start);
	
    // Setup
    if(clMatrixMultiplication.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    
    // Timing end.
	end_timing(&usage, &user_end, &system_end);
	
	// Get Time.
	getTiming_User_System(&read_time, &user_start, &user_end, &system_start, &system_end);
	
	// Write read time.
	writeReadTime(read_time);

	// Processing time start.
	gettimeofday(&time_start, NULL);
	
    // Run
    if(clMatrixMultiplication.setupCL()!=SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    
    if(clMatrixMultiplication.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    
    if(clMatrixMultiplication.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
	
	// Processing time end.
	gettimeofday(&time_end, NULL);
    getTiming(&process_time, &time_start, &time_end);
	
	// Write processing time.
	writeProcessTime(process_time);
	
	// Timing start.
	gettimeofday(&time_start, NULL);
    
    clMatrixMultiplication.writeOutput();
    
    // Timing end.
    gettimeofday(&time_end, NULL);
	
	// Get Time.
	getTiming(&write_time, &time_start, &time_end);
	
	// Write writing time.
	writeWriteTime(write_time);

    return 0;
}

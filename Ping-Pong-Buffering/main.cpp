#define _CRT_SECURE_NO_WARNINGS
#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define MATRIX_SIZE 1024
#define TILE_SIZE 16

// Helper function to check OpenCL API errors
void checkErr(cl_int err, const char* name) {
    if (err != CL_SUCCESS) {
        printf("OpenCL Error: %s failed with code %d\n", name, err);
        exit(1);
    }
}

// Load the OpenCL kernel text file
char* loadKernelSource(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open kernel file: %s\n", filename);
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (length <= 0) {
        fclose(file);
        printf("Kernel file is empty or invalid.\n");
        exit(1);
    }

    char* buffer = (char*)malloc((size_t)length + 1);
    if (!buffer) {
        fclose(file);
        printf("Failed to allocate memory for kernel source\n");
        exit(1);
    }
    size_t readSize = fread(buffer, 1, (size_t)length, file);
    buffer[readSize] = '\0';
    fclose(file);
    return buffer;
}

int main() {
    int N = MATRIX_SIZE;
    size_t matrixBytes = N * N * sizeof(float);

    printf("Initializing %dx%d matrices...\n", N, N);

    // Host Memory Allocation
    float* h_A = (float*)malloc(matrixBytes);
    float* h_B = (float*)malloc(matrixBytes);
    float* h_C_gpu = (float*)malloc(matrixBytes);

    if (!h_A || !h_B || !h_C_gpu) {
        printf("Failed to allocate host memory matrices.\n");
        exit(1);
    }

    srand((unsigned int)time(NULL));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Discover OpenCL Platform and Device
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(err, "clGetPlatformIDs count");

    cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
    if (!platforms) {
        printf("Failed to allocate memory for platforms.\n");
        exit(1);
    }
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    cl_device_id device = NULL;
    for (cl_uint i = 0; i < numPlatforms; ++i) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err == CL_SUCCESS) break;
    }

    if (!device) {
        printf("No GPU found, falling back to CPU...\n");
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }

    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("Using Device: %s\n", deviceName);

    // Create Context and Command Queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkErr(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkErr(err, "clCreateCommandQueue");

    // Create GPU VRAM Buffers
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixBytes, h_A, &err);
    checkErr(err, "clCreateBuffer A");

    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixBytes, h_B, &err);
    checkErr(err, "clCreateBuffer B");

    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrixBytes, NULL, &err);
    checkErr(err, "clCreateBuffer C");

    // Load and Build OpenCL Kernel
    char* kernelSource = loadKernelSource("matrix_kernel.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
    checkErr(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* buildLog = (char*)malloc(logSize);
        if (buildLog) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
            printf("\n--- BUILD ERROR ---\n%s\n-------------------\n", buildLog);
            free(buildLog);
        }
        exit(1);
    }

    // 6. Create Kernel and Pass Arguments
    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", &err);
    checkErr(err, "clCreateKernel");

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    // 7. Enqueue Kernel
    size_t globalWorkSize[2] = { (size_t)N, (size_t)N };
    size_t localWorkSize[2] = { TILE_SIZE, TILE_SIZE };

    printf("Executing Ping-Pong Tiled Kernel on GPU...\n");

    cl_event profEvent;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &profEvent);
    checkErr(err, "clEnqueueNDRangeKernel");

    clWaitForEvents(1, &profEvent);

    // Calculate Hardware Time
    cl_ulong timeStart, timeEnd;
    clGetEventProfilingInfo(profEvent, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, NULL);
    clGetEventProfilingInfo(profEvent, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, NULL);
    double executionTimeMs = (double)(timeEnd - timeStart) * 1e-6;

    printf("GPU Kernel Execution Time: %.3f ms\n", executionTimeMs);

    // Read back results
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, matrixBytes, h_C_gpu, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");

    // Quick Verification Check
    printf("Verifying sample cell output...\n");
    float expectedCell = 0.0f;
    for (int k = 0; k < N; ++k) {
        expectedCell += h_A[0 * N + k] * h_B[k * N + 0];
    }

    if (fabs(expectedCell - h_C_gpu[0]) < 1e-2f) {
        printf("SUCCESS: C[0][0] matches CPU calculation! (%.4f vs %.4f)\n", h_C_gpu[0], expectedCell);
    }
    else {
        printf("MISMATCH: GPU output = %.4f, CPU output = %.4f\n", h_C_gpu[0], expectedCell);
    }

    // Cleanup
    free(kernelSource);
    free(platforms);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    clReleaseEvent(profEvent);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
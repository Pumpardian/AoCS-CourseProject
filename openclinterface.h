#ifndef OPENCLINTERFACE_H
#define OPENCLINTERFACE_H

#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <CL/cl.h>

using namespace std;

class OpenCLInterface
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel histogram_kernel, prefix_sum_kernel, scatter_kernel;
    bool opencl_initialized = false;

    void initOpenCL();
    void cleanupOpenCL();

public:
    OpenCLInterface()
    {
        initOpenCL();
    }
    
    ~OpenCLInterface()
    {
        cleanupOpenCL();
    }

    void countSort_iGPU(vector<unsigned long long>& arr);
};

#endif //OPENCLINTERFACE_H
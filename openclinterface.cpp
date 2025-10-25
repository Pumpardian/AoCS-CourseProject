#include "openclinterface.h"

void OpenCLInterface::initOpenCL()
{
    cl_int ret;
    
    // Get platform and device
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &ret);

    // Pre-compile kernels
    const char* source = R"(
        __kernel void histogram(__global unsigned long long* arr, 
                                __global unsigned long long* count, 
                                unsigned long long size) {
            int i = get_global_id(0); 
            if (i < size) atomic_inc(&count[arr[i]]);
        }
        
        __kernel void prefix_sum(__global unsigned long long* count, 
                                unsigned long long max) {
            int i = get_global_id(0) + 1; 
            if (i <= max) count[i] += count[i-1];
        }
        
        __kernel void scatter(__global unsigned long long* arr,
                                __global unsigned long long* count, 
                                __global unsigned long long* output, 
                                unsigned long long size) {
            int i = get_global_id(0); 
            if (i < size) {
                unsigned long long value = arr[i];
                unsigned long long position = atomic_dec(&count[value]);
                output[position - 1] = value;
            }
        }
    )";

    program = clCreateProgramWithSource(context, 1, &source, NULL, &ret);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    // Create kernels once
    histogram_kernel = clCreateKernel(program, "histogram", &ret);
    prefix_sum_kernel = clCreateKernel(program, "prefix_sum", &ret);
    scatter_kernel = clCreateKernel(program, "scatter", &ret);
    
    opencl_initialized = true;
    printf("OpenCL initialized - kernels pre-compiled\n");

    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using device: %s\n", device_name);

    cl_uint max_compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
    printf("Compute units: %u\n", max_compute_units);

    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    printf("Max work group size: %zu\n", max_work_group_size);
}

void OpenCLInterface::cleanupOpenCL() 
{
    if (opencl_initialized)
    {
        clReleaseKernel(histogram_kernel);
        clReleaseKernel(prefix_sum_kernel);
        clReleaseKernel(scatter_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        opencl_initialized = false;
    }
}

void OpenCLInterface::countSort_iGPU(vector<unsigned long long>& arr)
{
    if (!opencl_initialized)
    {
        printf("OpenCL not initialized!\n");
        return;
    }
    
    unsigned long long max = *max_element(arr.begin(), arr.end());
    auto size = arr.size();
    vector<unsigned long long> output(size);
    
    cl_mem arr_buf, count_buf, output_buf;
    cl_int ret;

    // Create buffers
    arr_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                            size * sizeof(unsigned long long), arr.data(), &ret);
    count_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                (max + 1) * sizeof(unsigned long long), NULL, &ret);
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                size * sizeof(unsigned long long), NULL, &ret);

    // Initialize count buffer
    unsigned long long zero = 0;
    clEnqueueFillBuffer(queue, count_buf, &zero, sizeof(zero), 0, 
                        (max + 1) * sizeof(unsigned long long), 0, NULL, NULL);
    
    // Execute histogram kernel
    clSetKernelArg(histogram_kernel, 0, sizeof(cl_mem), &arr_buf);
    clSetKernelArg(histogram_kernel, 1, sizeof(cl_mem), &count_buf);
    clSetKernelArg(histogram_kernel, 2, sizeof(unsigned long long), &size);
    size_t global_size_hist = (size + 255) / 256 * 256; // Align to 256
    clEnqueueNDRangeKernel(queue, histogram_kernel, 1, NULL, &global_size_hist, NULL, 0, NULL, NULL);

    // Execute prefix sum kernel
    clSetKernelArg(prefix_sum_kernel, 0, sizeof(cl_mem), &count_buf);
    clSetKernelArg(prefix_sum_kernel, 1, sizeof(unsigned long long), &max);
    size_t global_size_prefix = (max + 255) / 256 * 256;
    clEnqueueNDRangeKernel(queue, prefix_sum_kernel, 1, NULL, &global_size_prefix, NULL, 0, NULL, NULL);

    // Execute scatter kernel
    clSetKernelArg(scatter_kernel, 0, sizeof(cl_mem), &arr_buf);
    clSetKernelArg(scatter_kernel, 1, sizeof(cl_mem), &count_buf);
    clSetKernelArg(scatter_kernel, 2, sizeof(cl_mem), &output_buf);
    clSetKernelArg(scatter_kernel, 3, sizeof(unsigned long long), &size);
    clEnqueueNDRangeKernel(queue, scatter_kernel, 1, NULL, &global_size_hist, NULL, 0, NULL, NULL);
    
    clFinish(queue); // Wait for all kernels to complete
    // Read results
    clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, 
                        size * sizeof(unsigned long long), output.data(), 0, NULL, NULL);

    // Release buffers only
    clReleaseMemObject(arr_buf);
    clReleaseMemObject(count_buf);
    clReleaseMemObject(output_buf);
}
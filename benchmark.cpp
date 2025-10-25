#include "benchmark.h"

vector<unsigned long long> Benchmark::generateArray(unsigned long long n)
{
    srand(0);

    vector<unsigned long long> arr(n);
    for (int i = 0; i < n; ++i)
    {
        arr[i] = rand() % 1000;
    }

    return arr;
}

void Benchmark::countSort(vector<unsigned long long>& arr)
{
    unsigned long long max = *max_element(arr.begin(), arr.end());
    auto size = arr.size();
    vector<unsigned long long> count(max + 1, 0);
    
    for (int i = 0; i < size; ++i)
    {
        ++count[arr[i]];
    }
    
    for (int i = 1; i <= max; ++i)
    {
        count[i] += count[i - 1];
    }
    
    vector<unsigned long long> output(size);
    for (int i = 0; i <= max; ++i)
    {
        int start = (i == 0) ? 0 : count[i - 1];
        int end = count[i];
        for (unsigned long long pos = start; pos < end; ++pos)
         {
            output[pos] = i;
        }
    }

    //arr = output;
}

void Benchmark::countSort_OMP(vector<unsigned long long>& arr)
{
    unsigned long long max = *max_element(arr.begin(), arr.end());
    auto size = arr.size();

    vector<unsigned long long> count(max + 1, 0);
    #pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        ++count[arr[i]];
    }
    
    for (int i = 1; i <= max; ++i)
    {
        count[i] += count[i - 1];
    }
    
    vector<unsigned long long> output(size);
    #pragma omp parallel for
    for (int i = 0; i <= max; ++i)
    {
        int start = (i == 0) ? 0 : count[i - 1];
        int end = count[i];
        for (unsigned long long pos = start; pos < end; ++pos)
        {
            output[pos] = i;
        }
    }

    //arr = output;
}

void Benchmark::countSort_OMP_iGPU(vector<unsigned long long>& arr)
{
    unsigned long long max = *max_element(arr.begin(), arr.end());
    auto size = arr.size();

    vector<unsigned long long> count(max + 1, 0);
    vector<unsigned long long> output(size);
    
    unsigned long long* arr_ptr = arr.data();
    unsigned long long* count_ptr = count.data();
    unsigned long long* output_ptr = output.data();
    
    #pragma omp target data map(to: arr_ptr[0:size]) \
                            map(alloc: count_ptr[0:max+1], output_ptr[0:size])
    {   
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < size; ++i)
        {
            ++count_ptr[arr_ptr[i]];
        }
        
        #pragma omp target teams distribute parallel for
        for (int i = 1; i <= max; ++i)
        {
            count_ptr[i] += count_ptr[i - 1];
        }
        
        #pragma omp target teams distribute parallel for
        for (int i = 0; i <= max; ++i)
        {
            int start = (i == 0) ? 0 : count_ptr[i - 1];
            int end = count_ptr[i];
            for (unsigned long long pos = start; pos < end; ++pos)
            {
                output_ptr[pos] = i;
            }
        }
        
        #pragma omp target update from(output_ptr[0:size])
    }
}

/*void Benchmark::countSort_iGPU(vector<unsigned long long>& arr)
{
    unsigned long long max = *max_element(arr.begin(), arr.end());
    auto size = arr.size();
    vector<unsigned long long> output(size);
    
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernels[3];
    cl_mem buffers[3];
    cl_int ret;

    // Setup
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &ret);

    // Buffers
    buffers[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                               size * sizeof(unsigned long long), arr.data(), &ret);
    buffers[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                               (max + 1) * sizeof(unsigned long long), NULL, &ret);
    buffers[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                               size * sizeof(unsigned long long), NULL, &ret);

    // Initialize count buffer to zero
    unsigned long long zero = 0;
    clEnqueueFillBuffer(queue, buffers[1], &zero, sizeof(zero), 0, 
                       (max + 1) * sizeof(unsigned long long), 0, NULL, NULL);

    // Optimized kernel source - PARALLEL scatter
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
        
        // OPTIMIZED: Parallel scatter using atomic operations
        __kernel void scatter_optimized(__global unsigned long long* arr,
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
        
        // ALTERNATIVE: Even faster for small max values
        __kernel void scatter_parallel(__global unsigned long long* count, 
                                      __global unsigned long long* output, 
                                      unsigned long long max,
                                      unsigned long long size) {
            int i = get_global_id(0); 
            if (i <= max) {
                unsigned long long start = (i == 0) ? 0 : count[i-1];
                unsigned long long end = count[i];
                unsigned long long count_val = end - start;
                
                // Parallel fill for this value
                for (unsigned long long j = get_global_id(0); j < count_val; j += get_global_size(0)) {
                    output[start + j] = i;
                }
            }
        }
    )";

    // Build program
    program = clCreateProgramWithSource(context, 1, &source, NULL, &ret);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    const char* kernel_names[] = {"histogram", "prefix_sum", "scatter_parallel"};
    size_t global_sizes[] = {size, max, size};  // All kernels now fully parallel
    
    for (int i = 0; i < 3; i++) {
        kernels[i] = clCreateKernel(program, kernel_names[i], &ret);
        
        if (i == 0) { // histogram
            clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &buffers[0]);
            clSetKernelArg(kernels[i], 1, sizeof(cl_mem), &buffers[1]);
            clSetKernelArg(kernels[i], 2, sizeof(unsigned long long), &size);
        }
        else if (i == 1) { // prefix_sum
            clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &buffers[1]);
            clSetKernelArg(kernels[i], 1, sizeof(unsigned long long), &max);
        }
        else { // scatter_optimized
            clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &buffers[0]);
            clSetKernelArg(kernels[i], 1, sizeof(cl_mem), &buffers[1]);
            clSetKernelArg(kernels[i], 2, sizeof(cl_mem), &buffers[2]);
            clSetKernelArg(kernels[i], 3, sizeof(unsigned long long), &size);
        }
        
        clEnqueueNDRangeKernel(queue, kernels[i], 1, NULL, &global_sizes[i], NULL, 0, NULL, NULL);
    }

    // Get results
    clEnqueueReadBuffer(queue, buffers[2], CL_TRUE, 0, 
                       size * sizeof(unsigned long long), output.data(), 0, NULL, NULL);

    // Cleanup
    for (auto k : kernels) clReleaseKernel(k);
    for (auto b : buffers) clReleaseMemObject(b);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}*/

void Benchmark::countSort_iGPU(vector<unsigned long long>& arr)
{
    openCLInterface->countSort_iGPU(arr);
}

long long Benchmark::test(void(Benchmark::*solution)(vector<unsigned long long>&), int size, int iterations, bool debug)
{
    vector<unsigned long long> arr = generateArray(size);

    /*if (debug)
    {
        for (auto& e : arr)
        {
            cout << e << ' ';
        }
        cout << '\n';
    }*/

    long long sum = 0;
    for (int i = 0; i < iterations; ++i)
    {
        auto start_time = chrono::high_resolution_clock::now();
        (this->*solution)(arr);
        auto end_time = chrono::high_resolution_clock::now();

        sum += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
    }

    /*if (debug)
    {
        for (auto& e : arr)
        {
            cout << e << ' ';
        }
        cout << '\n';
    }*/
    
    return sum / iterations;
}

void Benchmark::Run(int mode)
{
    if (testValues.empty())
    {
        for (long long n = 2; n < 100; n += 1)
        {
            testValues.push_back(n);
        }
        for (long long n = 100; n < 1000; n += 10)
        {
            testValues.push_back(n);
        }
        for (long long n = 1000; n < 10000; n += 100)
        {
            testValues.push_back(n);
        }
        for (long long n = 10000; n < 100000; n += 1000)
        {
            testValues.push_back(n);
        }
        for (long long n = 100000; n < 1000000; n += 10000)
        {
            testValues.push_back(n);
        }
        for (long long n = 1000000; n <= 5000000; n += 100000)
        {
            testValues.push_back(n);
        }
    }

    void (Benchmark::*function)(vector<unsigned long long>&);
    vector<long long> *results;
    switch (mode)
    {
        case BenchmarkMode::SingleCore:
            function = &Benchmark::countSort;
            results = &singleCoreResults;
            break;
        case BenchmarkMode::MultiCore:
            function = &Benchmark::countSort_OMP;
            results = &multiCoreResults;
            break;
        case BenchmarkMode::iGPU:
            function = &Benchmark::countSort_iGPU;
            results = &iGPUResults;
            break;
        default:
            return;
    }

    results->clear();
    long long total = testValues.size();
    for (long long i = 0; i < total;)
    {
        auto duration = test(function, testValues[i], ITERATIONS);
        results->emplace_back(duration);

        emit OnProgressUpdate((double)++i * 100 / total);
    }
}

void Benchmark::ExportResults(string filename)
{
    bool hasSCResults = !singleCoreResults.empty(), hasMCResults = !multiCoreResults.empty(), hasiGPUResults = !iGPUResults.empty();
    if (!(hasSCResults || hasMCResults || hasiGPUResults))
    {
        QMessageBox::information(nullptr, "No Data", "No results yet, nothing to write", QMessageBox::Ok);
        return;
    }

    ofstream file(filename);
    if (!file.is_open())
    {
        QMessageBox::critical(nullptr, "File Error", "Unable to open the file", QMessageBox::Ok);
        return;
    }

    file << 'n' << (hasSCResults ? ",SingleCore" : "")
     << (hasMCResults ? ",MultiCore" : "")
     << (hasiGPUResults ? ",iGPU" : "") << '\n';

    for (int i = 0; i < testValues.size(); ++i)
    {
        file << std::to_string(testValues[i]) << (hasSCResults ? ',' + std::to_string(singleCoreResults[i]) : "")
         << (hasMCResults ? ',' + std::to_string(multiCoreResults[i]) : "")
         << (hasiGPUResults ? ',' + std::to_string(iGPUResults[i]) : "") << '\n';
    }
    
    QMessageBox::information(nullptr, "Success", "Results have been written to the file", QMessageBox::Ok);
}

tuple<vector<long long> *, vector<long long> *, vector<long long> *, vector<long long> *> Benchmark::GetResults()
{
    return tuple<vector<long long> *, vector<long long> *, vector<long long> *, vector<long long> *>(&testValues, &singleCoreResults, &multiCoreResults, &iGPUResults);
}
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
        for (long long n = 1000000; n <= 2000000; n += 100000)
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
            function = &Benchmark::countSort_OMP_iGPU;
            results = &iGPUResults;
            break;
        default:
            return;
    }

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
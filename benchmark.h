#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <openclinterface.h>
#include <iomanip>
#include <fstream>
#include <omp.h>

#include <QMessageBox>
#define ITERATIONS 50

enum BenchmarkMode
{
    SingleCore,
    MultiCore,
    iGPU
};

class Benchmark : public QObject
{
    Q_OBJECT

    OpenCLInterface *openCLInterface;

    vector<long long> testValues;
    vector<long long> singleCoreResults;
    vector<long long> multiCoreResults;
    vector<long long> iGPUResults;

    vector<unsigned long long> generateArray(unsigned long long n);
    
    void countSort(vector<unsigned long long>& arr);
    void countSort_OMP(vector<unsigned long long>& arr);
    void countSort_iGPU(vector<unsigned long long>& arr);
    
    long long test(void(Benchmark::*solution)(vector<unsigned long long>&), int size, int iterations);
    
public:
    Benchmark()
    {
        openCLInterface = new OpenCLInterface();
		
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

    ~Benchmark()
    {
        delete openCLInterface;
        openCLInterface = nullptr;
    }

    void Run(int mode);
    void ExportResults(string filename);
    tuple<vector<long long>*, vector<long long>*, vector<long long>*, vector<long long>*> GetResults();

signals:
    double OnProgressUpdate(double progress);
};

#endif // BENCHMARK_H
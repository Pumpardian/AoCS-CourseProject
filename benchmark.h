#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <omp.h>

#include <QMessageBox>

using namespace std;
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

    vector<long long> testValues;
    vector<long long> singleCoreResults;
    vector<long long> multiCoreResults;
    vector<long long> iGPUResults;

    vector<unsigned long long> generateArray(unsigned long long n);
    
    void countSort(vector<unsigned long long>& arr);
    void countSort_OMP(vector<unsigned long long>& arr);
    void countSort_OMP_iGPU(vector<unsigned long long>& arr);
    
    long long test(void(Benchmark::*solution)(vector<unsigned long long>&), int size, int iterations, bool debug = false);
    
public:
    void Run(int mode);
    void ExportResults(string filename);
    tuple<vector<long long>*, vector<long long>*, vector<long long>*, vector<long long>*> GetResults();

signals:
    double OnProgressUpdate(double progress);
};

#endif // BENCHMARK_H
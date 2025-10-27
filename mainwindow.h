#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCharts>
#include "benchmark.h"
#include <QtConcurrent/QtConcurrent>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_Start_clicked();
    void on_pushButton_Export_clicked();
    void onBenchmarkFinished();
    void onProgressUpdated(double progress);

private:
    Ui::MainWindow *ui;
    QChartView *chartView;
    Benchmark *benchmark;
    QFutureWatcher<void> futureWatcher;

    void setupChartView();
    
    void updateTable();
    void updateChart();
};

#endif // MAINWINDOW_H
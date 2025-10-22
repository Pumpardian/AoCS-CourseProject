#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->radioButton_SingleCore->setChecked(true);
  
    benchmark = new Benchmark();
    connect(benchmark, &Benchmark::OnProgressUpdate,
        this, &MainWindow::onProgressUpdated);

    setupChartView();
    ui->tableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setupChartView()
{
    chartView = new QChartView(new QChart());
    chartView->setRenderHint(QPainter::Antialiasing);
    ui->verticalLayout_6->addWidget(chartView);
}

void MainWindow::updateChart()
{
    QChart *chart = new QChart();
    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);

    QLineSeries *seriesSC = new QLineSeries();
    seriesSC->setName("Single Core");
    QLineSeries *seriesMC = new QLineSeries();
    seriesMC->setName("Multi Core");
    QLineSeries *seriesiGPU = new QLineSeries();
    seriesiGPU->setName("iGPU");

    auto [n, sc, mc, igpu] = benchmark->GetResults();
    if (!sc->empty())
    {
        for (int i = 0; i < n->size(); ++i)
        {
            seriesSC->append(n->at(i), sc->at(i));
        }
        chart->addSeries(seriesSC);
    }
    if (!mc->empty())
    {
        for (int i = 0; i < n->size(); ++i)
        {
            seriesMC->append(n->at(i), mc->at(i));
        }
        chart->addSeries(seriesMC);
    }
    if (!igpu->empty())
    {
        for (int i = 0; i < n->size(); ++i)
        {
            seriesiGPU->append(n->at(i), igpu->at(i));
        }
        chart->addSeries(seriesiGPU);
    }

    QValueAxis *xAxis = new QValueAxis();
    QValueAxis *yAxis = new QValueAxis();
    
    xAxis->setTitleText("Array size (n)");
    yAxis->setTitleText("Time (microseconds)");
    xAxis->setLabelFormat("%d");
    yAxis->setLabelFormat("%d");
    xAxis->setTickType(QValueAxis::TicksFixed);
    xAxis->setTickInterval(100000);
    xAxis->setTickCount(20);
    yAxis->setTickCount(10);

    chart->addAxis(xAxis, Qt::AlignBottom);
    chart->addAxis(yAxis, Qt::AlignLeft);

    if (!sc->empty()) seriesSC->attachAxis(xAxis);
    if (!sc->empty()) seriesSC->attachAxis(yAxis);
    if (!mc->empty()) seriesMC->attachAxis(xAxis);
    if (!mc->empty()) seriesMC->attachAxis(yAxis);
    if (!igpu->empty()) seriesiGPU->attachAxis(xAxis);
    if (!igpu->empty()) seriesiGPU->attachAxis(yAxis);

    chart->setTitle("Benchmark Results");

    delete chartView;
    ui->verticalLayout_6->removeWidget(chartView);
    chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    ui->verticalLayout_6->addWidget(chartView);
}

void MainWindow::updateTable()
{
    ui->tableWidget->clear();
    ui->tableWidget->setColumnCount(4);
    ui->tableWidget->setHorizontalHeaderLabels({"Test Size (n)", "Single Core (µs)", "Multi Core (µs)", "iGPU (µs)"});
    
    auto [n, sc, mc, igpu] = benchmark->GetResults();
    
    int rowCount = n->size();
    ui->tableWidget->setRowCount(rowCount);
    
    for (int row = 0; row < rowCount; ++row)
    {
        ui->tableWidget->setItem(row, 0, new QTableWidgetItem(QString::number(n->at(row))));
        
        if (row < sc->size())
        {
            ui->tableWidget->setItem(row, 1, new QTableWidgetItem(QString::number(sc->at(row))));
        }
        else
        {
            ui->tableWidget->setItem(row, 1, new QTableWidgetItem("N/A"));
        }
          
        if (row < mc->size())
        {
            ui->tableWidget->setItem(row, 2, new QTableWidgetItem(QString::number(mc->at(row))));
        }
        else
        {
            ui->tableWidget->setItem(row, 2, new QTableWidgetItem("N/A"));
        }
        
        if (row < igpu->size())
        {
            ui->tableWidget->setItem(row, 3, new QTableWidgetItem(QString::number(igpu->at(row))));
        }
        else
        {
            ui->tableWidget->setItem(row, 3, new QTableWidgetItem("N/A"));
        }
    }

    ui->tableWidget->resizeColumnsToContents();
}

void MainWindow::on_pushButton_Start_clicked()
{
    QThread* thread = new QThread;
    QFuture<void> future;

    if (ui->radioButton_SingleCore->isChecked())
    {
        future = QtConcurrent::run([this]()
        {
            return benchmark->Run(0);
        });
    }
    else if (ui->radioButton_MultiCore->isChecked())
    {
        future = QtConcurrent::run([this]()
        {
            return benchmark->Run(1);
        });
    }
    else if (ui->radioButton_iGPU->isChecked())
    {
        future = QtConcurrent::run([this]()
        {
            return benchmark->Run(2);
        });
    }
    else
    {
        QMessageBox::critical(nullptr, "Invalid input", "Invalid benchmark mode selected, how could this happen?", QMessageBox::Ok);
        return;
    }

    ui->pushButton_Start->setEnabled(false);
    ui->pushButton_Export->setEnabled(false);
    connect(&futureWatcher, &QFutureWatcher<void>::finished,
            this, &MainWindow::onBenchmarkFinished);
    
    futureWatcher.setFuture(future);
}

void MainWindow::on_pushButton_Export_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(nullptr, tr("Export"), "results.csv", tr("CSV (*.csv)"));
    benchmark->ExportResults(fileName.toStdString());
}

void MainWindow::onBenchmarkFinished()
{
    updateTable();
    updateChart();

    ui->pushButton_Start->setEnabled(true);
    ui->pushButton_Export->setEnabled(true);
}

void MainWindow::onProgressUpdated(double progress)
{
    ui->progressBar->setValue(progress);
}
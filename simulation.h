#ifndef SIMULATION_H
#define SIMULATION_H

#include <QDialog>

namespace Ui {
    class Simulation;
}

class Simulation : public QDialog
{
    Q_OBJECT
    QString dir;
public:
    explicit Simulation(QWidget *parent,QString dir_);
    ~Simulation();

private:
    Ui::Simulation *ui;

private slots:
    void on_pushButton_clicked();
    void on_buttonBox_accepted();
};

#endif // SIMULATION_H

#include <QMessageBox>
#include <QFileDialog>
#include <QMainWindow>
#include "match_db.h"
#include "ui_match_db.h"

match_db::match_db(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc_) :
    QDialog(parent),vbc(vbc_),
    ui(new Ui::match_db)
{
    ui->setupUi(this);
    on_match_consecutive_clicked();
    if(!tipl::contains(vbc->handle->db.index_list,"iso"))
        ui->normalize_iso->hide();
}

match_db::~match_db()
{
    delete ui;
}

void match_db::show_match_table(void)
{
    ui->match_table->clear();
    ui->match_table->setRowCount(vbc->handle->db.match.size());
    ui->match_table->setColumnCount(2);
    ui->match_table->setColumnWidth(0,150);
    ui->match_table->setColumnWidth(1,150);
    ui->match_table->setHorizontalHeaderLabels(QStringList() << "Scan 1" << "Scan 2");
    for(int i = 0;i < vbc->handle->db.match.size();++i)
    {
        ui->match_table->setItem(i,0,
            new QTableWidgetItem(QString(vbc->handle->db.subject_names[vbc->handle->db.match[i].first].c_str())));
        ui->match_table->setItem(i,1,
            new QTableWidgetItem(QString(vbc->handle->db.subject_names[vbc->handle->db.match[i].second].c_str())));
    }
}


void match_db::on_buttonBox_accepted()
{
    if(vbc->handle->db.match.empty())
    {
        QMessageBox::critical(this,"ERROR","Match data first before calculating change");
        accept();
    }
    unsigned char dif_type = 0;
    if(ui->dif_type1->isChecked())
        dif_type = 0;
    if(ui->dif_type2->isChecked())
        dif_type = 1;
    vbc->handle->db.calculate_change(dif_type,ui->inc_dec_filter->currentIndex(),ui->normalize_iso->isChecked());
    QMessageBox::information(this,QApplication::applicationName(),"database updated");
    accept();
}

void match_db::on_load_match_clicked()
{
    QString FileName = QFileDialog::getOpenFileName(
                                this,
                                "Open match text file",
                                QFileInfo(((QMainWindow*)parent())->windowTitle()).absoluteDir().absolutePath(),
                                "Text file (*.txt);;All files (*)");
    if(FileName.isEmpty())
        return;
    std::vector<size_t> data;
    std::ifstream in(FileName.toStdString().c_str());
    std::copy(std::istream_iterator<int>(in),
              std::istream_iterator<int>(),std::back_inserter(data));
    vbc->handle->db.match.clear();
    for(size_t i = 0 ;i+1 < data.size();i+= 2)
        vbc->handle->db.match.push_back(std::make_pair(data[i],data[i+1]));
    show_match_table();
}

void match_db::on_match_consecutive_clicked()
{
    vbc->handle->db.match.clear();
    for(size_t i = 0;i < vbc->handle->db.subject_names.size();i += 2)
        vbc->handle->db.match.push_back(std::make_pair(i,i+1));
    show_match_table();
}

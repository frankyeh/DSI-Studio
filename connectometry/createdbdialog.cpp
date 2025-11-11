#include <QFileDialog>
#include <QInputDialog>
#include <QStringListModel>
#include <QMessageBox>
#include <fstream>
#include "createdbdialog.h"
#include "ui_createdbdialog.h"
#include "fib_data.hpp"
#include "connectometry/group_connectometry_analysis.h"
#include "image_model.hpp"


CreateDBDialog::CreateDBDialog(QWidget *parent,bool create_db_) :
    QDialog(parent),
    create_db(create_db_),
    dir_length(0),
    ui(new Ui::CreateDBDialog)
{
    ui->setupUi(this);
    ui->group_list->setModel(new QStringListModel);
    ui->group_list->setSelectionModel(new QItemSelectionModel(ui->group_list->model()));
    if(!create_db)
    {
        setWindowTitle("Create template");
        ui->movedown->hide();
        ui->moveup->hide();
        ui->create_data_base->setText("Create template");
        ui->subject_list_group->setTitle("Select subject FIB files");
    }
}

CreateDBDialog::~CreateDBDialog()
{
    delete ui;
}

void CreateDBDialog::on_close_clicked()
{
    close();
}

void CreateDBDialog::update_list(void)
{
    dir_length = 0;
    for(size_t i = 0;i < group.size();)
        if(group[i].endsWith("db.fib.gz") || group[i].endsWith("db.fz") || group[i].endsWith("dz") ||
           (QFileInfo(group[0]).completeSuffix() != QFileInfo(group[i]).completeSuffix()))
            group.removeAt(i);
        else
            ++i;
    if(group.size() > 1)
    {
        QDir dir1(QFileInfo(group[0]).dir()),dir2(QFileInfo(group[group.size()-1]).dir());
        if(dir1.absolutePath() == dir2.absolutePath())
            dir_length = 0;
        else
        while(dir1.cdUp() && dir2.cdUp())
        {
            if(dir1.absolutePath() == dir2.absolutePath())
            {
                dir_length = dir1.absolutePath().length()+1;
                break;
            }
        }
    }    
    if(create_db && !group.empty() && sample_fib != group[0])
    {
        sample_fib = group[0];
        if(create_db)
        {
            fib_data fib;
            if(!fib.load_from_file(sample_fib.toStdString()))
            {
                QMessageBox::critical(this,"ERROR",fib.error_msg.c_str());
                raise(); // for Mac
                return;
            }
            template_reso = fib.vs[0];
            template_id = fib.template_id;
        }
    }
    if(ui->output_file_name->text().isEmpty())
        update_output_file_name();

    QStringList filenames;
    for(unsigned int index = 0;index < group.size();++index)
        filenames << QFileInfo(group[index]).baseName();
    ((QStringListModel*)ui->group_list->model())->setStringList(filenames);

    raise(); // for Mac
}

void CreateDBDialog::on_group1open_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                     this,"Open Files","",
                                     create_db ?
                                     "Fib Files (*.fz *fib.gz);;All Files (*)" :
                                     "Fib Files (*.fz *fib.gz);;NIFTI Files (*nii *nii.gz);;All Files (*)");
    if (filenames.isEmpty())
        return;
    group << filenames;
    update_list();
    update_output_file_name();
}

void CreateDBDialog::on_group1delete_clicked()
{
    QModelIndexList indexes = ui->group_list->selectionModel()->selectedRows();
    if(!indexes.count())
        return;
    group.erase(group.begin()+indexes.first().row());
    update_list();
}


void CreateDBDialog::on_moveup_clicked()
{
    QModelIndexList indexes = ui->group_list->selectionModel()->selectedRows();
    if(!indexes.count() || indexes.first().row() == 0)
        return;
    #ifdef QT6_PATCH
        group.swapItemsAt(indexes.first().row(),indexes.first().row()-1);
    #else
        group.swap(indexes.first().row(),indexes.first().row()-1);
    #endif
    update_list();
    ui->group_list->selectionModel()->select(ui->group_list->model()->index(indexes.first().row()-1,0),
                                             QItemSelectionModel::Select);
}

void CreateDBDialog::on_movedown_clicked()
{
    QModelIndexList indexes = ui->group_list->selectionModel()->selectedRows();
    if(!indexes.count() || indexes.first().row() == group.size()-1)
        return;
    #ifdef QT6_PATCH
        group.swapItemsAt(indexes.first().row(),indexes.first().row()+1);
    #else
        group.swap(indexes.first().row(),indexes.first().row()+1);
    #endif
    update_list();
    ui->group_list->selectionModel()->select(ui->group_list->model()->index(indexes.first().row()+1,0),
                                             QItemSelectionModel::Select);
}

void CreateDBDialog::on_sort_clicked()
{
    if(group.empty())
        return;
    group.sort();
    update_list();
}


void CreateDBDialog::on_open_list1_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                                 this,
                                 "Open text file",
                                 "",
                                 "Text files (*.txt);;All files (*)" );
    if(filename.isEmpty())
        return;
    group.clear();
    for(const auto& line: tipl::read_text_file(filename.toStdString()))
        group << line.c_str();
    update_list();
}

void CreateDBDialog::on_save_list1_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                                 this,
                                 "Open text file",
                                 "",
                                 "Text files (*.txt);;All files (*)" );
    if(filename.isEmpty())
        return;

    std::ofstream out(filename.toStdString());
    for(int index = 0;index < group.size();++index)
        out << group[index].toStdString().c_str() <<  std::endl;
}

QStringList search_files(QString dir,QString filter);
void CreateDBDialog::on_open_dir1_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                "");
    if(dir.isEmpty())
        return;
    group << search_files(dir,"*.fz");
    update_list();
}

void CreateDBDialog::on_select_output_file_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                                 this,
                                 "Save file",
                                 "","FIB files (*.fz *fib.gz);;NIFTI Files(*nii.gz);;All files (*)");
    if(filename.isEmpty())
        return;
#ifdef __APPLE__
// fix the Qt double extension bug here
    if(QFileInfo(filename).completeSuffix().contains("fib.gz"))
        filename = QFileInfo(filename).absolutePath() + "/" + QFileInfo(filename).baseName() + "fib.gz";
#endif
    ui->output_file_name->setText(filename);
}
bool odf_average(const char* out_name,std::vector<std::string>& file_names,std::string& error_msg);
void CreateDBDialog::on_create_data_base_clicked()
{
    if(ui->output_file_name->text().isEmpty())
    {
        QMessageBox::critical(this,"ERROR","Please assign output file");
        return;
    }
    if(group.empty())
    {
        QMessageBox::critical(this,"ERROR","Please assign subject files");
        return;
    }

    if(create_db)
    {
        std::vector<std::string> name_list;
        for(auto each : group)
            name_list.push_back(each.toStdString());

        fib_data fib;
        if(!fib.load_template_fib(template_id,template_reso) ||
           !fib.db.create_db(name_list) ||
           !fib.save_to_file(ui->output_file_name->text().toStdString()))
        {
            if(!fib.error_msg.empty())
                QMessageBox::critical(this,"ERROR",fib.error_msg.c_str());
        }
        else
            QMessageBox::information(this,QApplication::applicationName(),"database created");
    }
    else
    {
        std::vector<std::string> name_list(group.count());
        for (unsigned int index = 0;index < group.count();++index)
            name_list[index] = group[index].toStdString().c_str();

        std::string error_msg;
        if(!odf_average(ui->output_file_name->text().toStdString().c_str(),name_list,error_msg))
        {
            if(!error_msg.empty())
                QMessageBox::critical(this,"ERROR",error_msg.c_str());
        }
        else
            QMessageBox::information(this,QApplication::applicationName(),"File created");
    }
    raise(); // for Mac
}



void CreateDBDialog::update_output_file_name(void)
{
    if(!group.empty())
    {
        std::string front = group.front().toStdString();
        std::string back = group.back().toStdString();
        QString base_name = std::string(front.begin(),
                        std::mismatch(front.begin(),front.begin()+
                        int64_t(std::min(front.length(),back.length())),back.begin()).first).c_str();
        if(create_db)
            ui->output_file_name->setText(base_name + ".dz");
        else
        {
            if(tipl::ends_with(front,".nii.gz") || tipl::ends_with(front,".nii"))
                ui->output_file_name->setText(base_name + ".nii.gz");
            else
                ui->output_file_name->setText(base_name + ".avg.fz");
        }
    }
}


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

extern std::vector<std::string> fib_template_list;
void populate_templates(QComboBox* combo,size_t index);
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
        ui->index_of_interest->hide();
        ui->template_widget->hide();
        ui->movedown->hide();
        ui->moveup->hide();
        ui->create_data_base->setText("Create template");
        ui->subject_list_group->setTitle("Select subject FIB files");
        ui->index_label->hide();
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

QString CreateDBDialog::get_file_name(QString file_path)
{
    if(dir_length)
    {
        for(int j = dir_length;j < file_path.length();++j)
            if(file_path[j] == '\\' || file_path[j] == '/' || file_path[j] == ' ')
                file_path[j] = '_';

    }
    return QFileInfo(file_path).baseName();
}

void CreateDBDialog::update_list(void)
{
    dir_length = 0;
    for(size_t i = 0;i < group.size();)
        if(group[i].endsWith("db.fib.gz"))
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
    if(!group.empty() && sample_fib != group[0])
    {
        sample_fib = group[0];

        if(group[0].endsWith("nii") || group[0].endsWith("nii.gz"))
        {
            bool ok;
            QString metrics = QInputDialog::getText(this,"DSI Studio","Please specify the name of the metrics",QLineEdit::Normal,"metrics",&ok);
            if(!ok)
                metrics = "metrics";
            ui->index_of_interest->clear();
            ui->index_of_interest->addItem(metrics);
            ui->index_of_interest->setCurrentIndex(0);
            populate_templates(ui->template_list,0);
            ui->template_list->setEnabled(true);
        }
        else
        {
            fib_data fib;
            if(!fib.load_from_file(sample_fib.toStdString().c_str()))
            {
                QMessageBox::critical(this,"ERROR","The first file is not a valid FIB file.");
                raise(); // for Mac
                return;
            }
            fib_reso = std::floor((fib.vs[0] + fib.vs[2])*0.5f*100.0f)/100.0f;
            ui->index_of_interest->clear();
            std::vector<std::string> item_list;
            fib.get_index_list(item_list);
            for(auto& name : item_list)
                ui->index_of_interest->addItem(name.c_str());
            populate_templates(ui->template_list,fib.template_id);
            ui->template_list->setEnabled(!fib.is_mni);
        }
        ui->template_list->addItem("Open...");
    }
    if(ui->output_file_name->text().isEmpty())
        on_index_of_interest_currentTextChanged(QString());

    QStringList filenames;
    for(unsigned int index = 0;index < group.size();++index)
        filenames << get_file_name(group[index]);
    ((QStringListModel*)ui->group_list->model())->setStringList(filenames);

    raise(); // for Mac
}

void CreateDBDialog::on_group1open_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                     this,
                                     "Open Fib files",
                                     "",
                                     create_db ? "Fib Files (*fib.gz);;NIFTI Files (*nii *nii.gz);;All Files (*)":
                                                 "Fib Files (*fib.gz);;All Files (*)");
    if (filenames.isEmpty())
        return;
    group << filenames;
    update_list();
    on_index_of_interest_currentTextChanged(QString());
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
    if(QFileInfo(group[0]).baseName().count('_') == 2)
    {
        std::map<QString,QString> sort_map;
        for(unsigned int index = 0;index < group.size();++index)
        {
            QString str = QFileInfo(group[index]).baseName();
            int pos = str.lastIndexOf('_')+1;
            sort_map[pos ? str.right(str.length()-pos):str] = group[index];
        }
        std::vector<std::pair<QString,QString> > sorted_groups(sort_map.begin(),sort_map.end());
        for(unsigned int index = 0;index < sorted_groups.size();++index)
            group[index] = sorted_groups[index].second;
    }
    else
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
    std::string line;
    std::ifstream in(filename.toStdString().c_str());
    while(std::getline(in,line))
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

    std::ofstream out(filename.toStdString().c_str());
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
    group << search_files(dir,"*.fib.gz");
    update_list();
}

void CreateDBDialog::on_select_output_file_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                                 this,
                                 "Save file",
                                 "",
                                 "FIB file (*fib.gz);;All files (*)");
    if(filename.isEmpty())
        return;
#ifdef __APPLE__
// fix the Qt double extension bug here
    if(QFileInfo(filename).completeSuffix().contains("fib.gz"))
        filename = QFileInfo(filename).absolutePath() + "/" + QFileInfo(filename).baseName() + "fib.gz";
#endif
    ui->output_file_name->setText(filename);
}

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
        std::string template_file_name;
        auto template_id = ui->template_list->currentIndex();
        if(template_id >= fib_template_list.size())
        {
            template_file_name = QFileDialog::getOpenFileName(
                                             this,
                                             "Open Template FIB File",
                                             "","Fib Files (*fib.gz);;All Files (*)").toStdString();
            if(template_file_name.empty())
                return;
        }
        else
        {
            if(template_id == -1 || fib_template_list[template_id].empty())
            {
                QMessageBox::critical(this,"ERROR","Cannot find the template for creating database");
                return;
            }
            template_file_name = fib_template_list[template_id];
        }
        std::shared_ptr<fib_data> template_fib;
        template_fib.reset(new fib_data);
        if(!template_fib->load_from_file(template_file_name.c_str()) ||
           (fib_reso > template_fib->vs[0] && !template_fib->resample_to(fib_reso)))
            {
                QMessageBox::critical(this,"ERROR",template_fib->error_msg.c_str());
                return;
            }

        tipl::progress prog_("creating database");
        std::shared_ptr<group_connectometry_analysis> data(new group_connectometry_analysis);

        if(!data->create_database(template_fib))
        {
            QMessageBox::critical(this,"ERROR",data->error_msg.c_str());
            return;
        }
        if(!data->handle->is_mni)
        {
            QMessageBox::critical(this,"ERROR","the template has to be QSDR reconstructed FIB file");
            return;
        }
        data->handle->db.index_name = ui->index_of_interest->currentText().toStdString();

        tipl::progress prog("reading data");
        for (unsigned int index = 0;prog(index,group.count());++index)
        {
            tipl::out() << QFileInfo(group[index]).baseName().toStdString() << std::endl;
            if(!data->handle->db.add_subject_file(group[index].toStdString(),get_file_name(group[index]).toStdString()))
            {
                QMessageBox::critical(this,"ERROR",data->handle->db.error_msg.c_str());
                raise(); // for Mac
                return;
            }
        }
        if(prog.aborted())
            return;
        if(!data->handle->db.save_db(ui->output_file_name->text().toStdString().c_str()))
            QMessageBox::critical(this,"ERROR",data->handle->db.error_msg.c_str());
        else
            QMessageBox::information(this,"Connectometry database created",ui->output_file_name->text());
    }
    else
    {
        std::vector<std::string> name_list(group.count());
        for (unsigned int index = 0;index < group.count();++index)
            name_list[index] = group[index].toStdString().c_str();
        const char* error_msg = odf_average(ui->output_file_name->text().toStdString().c_str(),name_list);
        if(error_msg)
            QMessageBox::critical(this,"ERROR",error_msg);
        else
            QMessageBox::information(this,"completed","File created");
    }
    raise(); // for Mac
}



void CreateDBDialog::on_index_of_interest_currentTextChanged(const QString &arg1)
{
    if(!group.empty())
    {
        std::string front = group.front().toStdString();
        std::string back = group.back().toStdString();
        QString base_name = std::string(front.begin(),
                        std::mismatch(front.begin(),front.begin()+
                        int64_t(std::min(front.length(),back.length())),back.begin()).first).c_str();

        if(create_db)
            ui->output_file_name->setText(base_name + "." + ui->index_of_interest->currentText() + ".db.fib.gz");
        else
            ui->output_file_name->setText(base_name + ".avg.fib.gz");
    }
}


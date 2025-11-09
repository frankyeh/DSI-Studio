#include <QFileInfo>
#include <QInputDialog>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QGraphicsTextItem>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
#include "db_window.h"
#include "ui_db_window.h"
#include "match_db.h"
#include "TIPL/tipl.hpp"

#include <filesystem>
QString check_citation(QString str);
db_window::db_window(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc_) :
    QMainWindow(parent),vbc(vbc_),
    ui(new Ui::db_window)
{
    ui->setupUi(this);
    ui->report->setText(check_citation(QString::fromStdString(vbc->handle->report)));
    ui->vbc_view->setScene(&vbc_scene);

    ui->x_pos->setMaximum(vbc->handle->dim[0]-1);
    ui->y_pos->setMaximum(vbc->handle->dim[1]-1);
    ui->z_pos->setMaximum(vbc->handle->dim[2]-1);

    connect(ui->slice_pos,SIGNAL(valueChanged(int)),this,SLOT(on_subject_list_itemSelectionChanged()));
    connect(ui->view_y,SIGNAL(toggled(bool)),this,SLOT(on_view_x_toggled(bool)));
    connect(ui->view_z,SIGNAL(toggled(bool)),this,SLOT(on_view_x_toggled(bool)));

    connect(ui->zoom,SIGNAL(valueChanged(double)),this,SLOT(on_subject_list_itemSelectionChanged()));

    on_view_x_toggled(true);
    for(const auto& each : vbc->handle->db.index_list)
        ui->index_name->addItem(QString::fromStdString(each));
    update_subject_list();
    ui->subject_list->selectRow(0);
    qApp->installEventFilter(this);


}

db_window::~db_window()
{
    qApp->removeEventFilter(this);
    delete ui;
}

void db_window::closeEvent (QCloseEvent *event)
{
    if(!vbc->handle->db.modified)
    {
        event->accept();
        return;
    }

    QMessageBox::StandardButton r = QMessageBox::question( this, QApplication::applicationName(),
                                                                tr("Modification not saved. Save now?\n"),
                                                                QMessageBox::Cancel | QMessageBox::No | QMessageBox::Yes,
                                                                QMessageBox::Cancel);
    if (r == QMessageBox::Cancel)
    {
        event->ignore();
        return;
    }
    if (r == QMessageBox::No)
    {
        event->accept();
        return;
    }
    if (r == QMessageBox::Yes)
    {
        on_actionSave_DB_as_triggered();
        if(!vbc->handle->db.modified)
        {
            event->accept();
            return;
        }
        else
        {
            event->ignore();
            return;
        }
    }
}


bool db_window::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() != QEvent::MouseMove || obj->parent() != ui->vbc_view)
        return false;
    QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
    QPointF point = ui->vbc_view->mapToScene(mouseEvent->pos().x(),mouseEvent->pos().y());
    tipl::vector<3> pos;
    pos[0] =  float(point.x()) / ui->zoom->value() - 0.5f;
    pos[1] =  float(point.y()) / ui->zoom->value() - 0.5f;
    pos[2] = ui->slice_pos->value();
    if(!vbc->handle->dim.is_valid(pos))
        return true;
    ui->x_pos->setValue(std::round(pos[0]));
    ui->y_pos->setValue(std::round(pos[1]));
    ui->z_pos->setValue(std::round(pos[2]));


    return true;
}

void db_window::update_subject_list()
{
    ui->subject_list->setCurrentCell(0,0);
    ui->subject_list->clear();
    ui->subject_list->setColumnCount(vbc->handle->db.titles.size() + 1);
    ui->subject_list->setColumnWidth(0,100);
    ui->subject_list->setRowCount(vbc->handle->db.subject_names.size());
    QStringList header;
    header << "id";
    for(auto& str : vbc->handle->db.titles)
        header << str.c_str();
    ui->subject_list->setHorizontalHeaderLabels(header);

    for(unsigned int i = 0;i < vbc->handle->db.subject_names.size();++i)
    {
        ui->subject_list->setItem(i,0, new QTableWidgetItem(QString(vbc->handle->db.subject_names[i].c_str())));
        for(unsigned int j = 0;j < vbc->handle->db.titles.size();++j)
            ui->subject_list->setItem(i,j+1, new QTableWidgetItem(QString(vbc->handle->db.items[i*vbc->handle->db.titles.size() + j].c_str())));
    }

}

void db_window::on_subject_list_itemSelectionChanged()
{
    if(ui->subject_list->currentRow() == -1 ||
            ui->subject_list->currentRow() >= vbc->handle->db.subject_names.size())
        return;
    if(ui->view_x->isChecked())
        ui->x_pos->setValue(ui->slice_pos->value());
    if(ui->view_y->isChecked())
        ui->y_pos->setValue(ui->slice_pos->value());
    if(ui->view_z->isChecked())
        ui->z_pos->setValue(ui->slice_pos->value());

    tipl::image<2,float> slice;
    vbc->handle->db.get_subject_slice(ui->subject_list->currentRow(),
                                   ui->view_x->isChecked() ? 0:(ui->view_y->isChecked() ? 1:2),
                                   ui->slice_pos->value(),slice);

    float m = tipl::max_abs_value(slice);
    if(m != 0.0f)
        m = 255.99f/m;
    tipl::color_image color_slice(slice.shape());
    for(size_t i = 0;i < color_slice.size();++i)
    {
        if(slice[i] < 0)
            color_slice[i].r = uint8_t(-slice[i]*m);
        else
            color_slice[i] = uint8_t(slice[i]*m);
    }
    vbc_slice_image << color_slice;
    vbc_slice_image = vbc_slice_image.scaled(color_slice.width()*ui->zoom->value(),color_slice.height()*ui->zoom->value());
    if(!ui->view_z->isChecked())
        vbc_slice_image = vbc_slice_image.mirrored();

    vbc_scene << vbc_slice_image;

    vbc_slice_pos = ui->slice_pos->value();

}

void db_window::on_actionSave_Subject_Name_as_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save name list",
                windowTitle() + ".name.txt",
                "Report file (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    std::ofstream out(filename.toStdString().c_str());
    for(const auto& each : vbc->handle->db.subject_names)
        out << each << std::endl;
}

void db_window::on_action_Save_R2_values_as_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save R2 values",
                windowTitle() + ".R2.txt",
                "Report file (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    std::ofstream out(filename.toStdString().c_str());
    std::copy(vbc->handle->db.R2.begin(),vbc->handle->db.R2.end(),std::ostream_iterator<float>(out,"\n"));
}

void db_window::on_view_x_toggled(bool checked)
{
    if(!checked)
        return;
    unsigned char dim = ui->view_x->isChecked() ? 0:(ui->view_y->isChecked() ? 1:2);
    ui->slice_pos->setMaximum(vbc->handle->dim[dim]-1);
    ui->slice_pos->setMinimum(0);
    ui->slice_pos->setValue(vbc->handle->dim[dim] >> 1);
}

void db_window::update_db(void)
{
    update_subject_list();
    ui->report->setText(check_citation(vbc->handle->report.c_str()));
}


void db_window::on_delete_subject_clicked()
{
    if(ui->subject_list->currentRow() >= 0 && vbc->handle->db.subject_names.size() > 1)
    {
        unsigned int index = ui->subject_list->currentRow();
        vbc->handle->db.remove_subject(index);
        if(index < ui->subject_list->rowCount())
            ui->subject_list->removeRow(index);
    }
}

void db_window::on_actionCalculate_change_triggered()
{
    if(vbc->handle->db.is_longitudinal)
    {
        QMessageBox::critical(this,"ERROR","The data cannot compute differences in longitudinal data");
        return;
    }
    std::unique_ptr<match_db> mdb(new match_db(this,vbc));
    if(mdb->exec() == QDialog::Accepted)
    {
        if(!vbc->handle->db.demo.empty())
            vbc->handle->db.parse_demo();
        update_db();
    }
}

void db_window::on_actionSave_DB_as_triggered()
{
    QString default_ext = ".mod.dz";
    if(vbc->handle->db.is_longitudinal)
    {
        default_ext = ".dif.dz";
        if(vbc->handle->db.longitudinal_filter_type == 1)
            default_ext = ".pos_dif.dz";
        if(vbc->handle->db.longitudinal_filter_type == 2)
            default_ext = ".neg_dif.dz";

    }
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save Database",
                           windowTitle()+default_ext,
                           "Database files (*.dz);;All files (*)");
    if (filename.isEmpty())
        return;
    tipl::progress prog_("saving ",std::filesystem::path(filename.toStdString()).filename().u8string().c_str());
    if(!vbc->handle->db.demo.empty() && !vbc->handle->db.parse_demo())
        QMessageBox::information(this,QApplication::applicationName(),
        QString("demographics not saved due to mismatch: ") + vbc->handle->error_msg.c_str());

    if(vbc->handle->save_to_file(filename.toStdString()))
        QMessageBox::information(this,QApplication::applicationName(),"File saved");
    else
        QMessageBox::critical(this,"ERROR",vbc->handle->error_msg.c_str());
}

void db_window::on_move_down_clicked()
{

    if(ui->subject_list->currentRow() >= vbc->handle->db.subject_names.size()-1)
        return;
    vbc->handle->db.move_down(ui->subject_list->currentRow());
    QString t = ui->subject_list->item(ui->subject_list->currentRow(),0)->text();
    ui->subject_list->item(ui->subject_list->currentRow(),0)->setText(ui->subject_list->item(ui->subject_list->currentRow()+1,0)->text());
    ui->subject_list->item(ui->subject_list->currentRow()+1,0)->setText(t);
    ui->subject_list->selectRow(ui->subject_list->currentRow()+1);
}

void db_window::on_move_up_clicked()
{
    if(ui->subject_list->currentRow() <= 0)
        return;
    vbc->handle->db.move_up(ui->subject_list->currentRow());
    QString t = ui->subject_list->item(ui->subject_list->currentRow(),0)->text();
    ui->subject_list->item(ui->subject_list->currentRow(),0)->setText(ui->subject_list->item(ui->subject_list->currentRow()-1,0)->text());
    ui->subject_list->item(ui->subject_list->currentRow()-1,0)->setText(t);
    ui->subject_list->selectRow(ui->subject_list->currentRow()-1);

}

void db_window::on_actionAdd_DB_triggered()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                           this,
                           "Select file to add",
                           windowTitle(),
                           "Database files (*.fz *fib.gz);;All files (*)");
    if (filenames.isEmpty())
        return;
    std::vector<std::string> file_names;
    for(auto each: filenames)
        file_names.push_back(each.toStdString());
    if(!vbc->handle->db.add_subjects(file_names) && !vbc->handle->error_msg.empty())
        QMessageBox::critical(this,"ERROR",vbc->handle->error_msg.c_str());
    update_db();
}

void db_window::on_actionAdd_Database_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,
                           "Select file to add",
                           windowTitle(),
                           "Database files (*.dz *.db.fib.gz *.db.fz);;All files (*)");
    if (filename.isEmpty())
        return;
    if(!vbc->handle->db.add_db(filename.toStdString()) && !vbc->handle->error_msg.empty())
        QMessageBox::critical(this,"ERROR",vbc->handle->error_msg.c_str());
    update_db();
}


void db_window::on_actionSelect_Subjects_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,
                           "Open Selection Text files",
                           QFileInfo(windowTitle()).absoluteDir().absolutePath(),
                           "Text files (*.txt);;All files (*)");
    if (filename.isEmpty())
        return;
    std::ifstream in(filename.toStdString().c_str());
    std::vector<char> selected;
    std::copy(std::istream_iterator<int>(in),std::istream_iterator<int>(),std::back_inserter(selected));
    selected.resize(vbc->handle->db.subject_names.size());
    for(int i = int(selected.size())-1;i >=0;--i)
        if(selected[uint32_t(i)])
        {
            vbc->handle->db.remove_subject(uint32_t(i));
            ui->subject_list->removeRow(i);
        }
}

void db_window::on_actionCurrent_Subject_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Open Selection Text files",
                           QFileInfo(windowTitle()).absoluteDir().absolutePath()+"\\"+
                            vbc->handle->db.subject_names[uint32_t(ui->subject_list->currentRow())].c_str()+"."+
                            vbc->handle->db.index_name.c_str()+".nii.gz",
                           "NIFTI files (*.nii *nii.gz);;All files (*)");
    if (filename.isEmpty())
        return;
    if(tipl::io::gz_nifti::save_to_file<tipl::progress,tipl::error>(filename.toStdString().c_str(),
                                                        vbc->handle->bind(
                    vbc->handle->db.get_index_image(uint32_t(ui->subject_list->currentRow())))))
        QMessageBox::information(this,QApplication::applicationName(),"file saved");
    else
        QMessageBox::critical(this,"ERROR","cannot save file.");
}

void db_window::on_actionAll_Subjects_triggered()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Output directory",
                                "");
    if(dir.isEmpty())
        return;
    tipl::progress prog("exporting ",dir.toStdString().c_str());
    for(size_t i = 0;prog(i,vbc->handle->db.subject_names.size());++i)
    {
        QString file_name = dir + "\\"+
                vbc->handle->db.subject_names[i].c_str()+"."+
                vbc->handle->db.index_name.c_str()+".nii.gz";
        tipl::image<3> I = vbc->handle->db.get_index_image(uint32_t(i));
        tipl::io::gz_nifti out;
        out.set_voxel_size(vbc->handle->vs);
        out.set_image_transformation(vbc->handle->trans_to_mni);
        out << I;
        if(!out.save_to_file(file_name.toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR","Cannot save file.");
            return;
        }
    }
    QMessageBox::information(this,QApplication::applicationName(),"Files exported");
}

void db_window::on_actionOpen_Demographics_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                this,
                "Open demographics",
                QFileInfo(vbc->handle->fib_file_name.c_str()).absoluteDir().absolutePath(),
                "Comma- or Tab-Separated Values(*.csv *.tsv);;All files (*)");
    if(filename.isEmpty())
        return;
    if(vbc->handle->db.parse_demo(filename.toStdString()))
    {
        QMessageBox::information(this,QApplication::applicationName(),"Demographics Loaded");
        update_subject_list();
    }
    else
        QMessageBox::critical(this,"ERROR",vbc->handle->error_msg.c_str());
}


void db_window::on_actionSave_Demographics_triggered()
{
    if(vbc->handle->db.demo.empty())
    {
        QMessageBox::critical(this,"ERROR","No demographic data in the database");
        return;
    }
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Save demographics",
                QFileInfo(vbc->handle->fib_file_name.c_str()).absoluteDir().absolutePath(),
                "Comma- or Tab-Separated Values(*.csv *.tsv);;All files (*)");
    if(filename.isEmpty())
        return;
    std::ofstream out(filename.toStdString().c_str());
    out << vbc->handle->db.demo;
}

QString get_matched_demo(QWidget *parent,std::shared_ptr<fib_data> handle)
{
    std::string demo_cap, demo_sample;
    for(auto str: handle->db.feature_titles)
    {
        demo_cap += str;
        demo_cap += " ";
    }
    std::ostringstream out;
    // X +1 to skip intercept
    for(size_t i = 0;i < handle->db.feature_location.size() && i+1 < handle->db.X.size();++i)
        out << handle->db.X[i+1] << " ";
    demo_sample = out.str();
    demo_sample.pop_back();

    bool ok;
    return QInputDialog::getText(parent,"Specify demographics",
        QString("Input demographic values for %1 separated by space").arg(demo_cap.c_str()),
                                          QLineEdit::Normal,demo_sample.c_str(),&ok);
}
void db_window::on_actionSave_DemoMatched_Image_as_triggered()
{
    if(vbc->handle->db.demo.empty())
    {
        QMessageBox::critical(this,"ERROR","No demographic data in the database");
        return;
    }

    QString param = get_matched_demo(this,vbc->handle);
    if (param.isEmpty())
                return;

    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Open Selection Text files",
                           QFileInfo(windowTitle()).absoluteDir().absolutePath()+"\\"+
                           vbc->handle->fib_file_name.c_str()+"."+QString(param).replace(' ','_').replace(',','_')+"."+
                           vbc->handle->db.index_name.c_str()+".nii.gz",
                           "NIFTI files (*.nii *nii.gz);;All files (*)");
    if (filename.isEmpty())
        return;
    if(vbc->handle->db.save_demo_matched_image(param.toStdString(),filename.toStdString()))
        QMessageBox::information(this,QApplication::applicationName(),"File Saved");
    else
        QMessageBox::critical(this,"ERROR",vbc->handle->error_msg.c_str());

}


void db_window::on_index_name_currentIndexChanged(int index)
{
    if(ui->index_name->currentText().toStdString() != vbc->handle->db.index_name)
    {
        vbc->handle->db.set_current_index(index);
        on_subject_list_itemSelectionChanged();
    }
}



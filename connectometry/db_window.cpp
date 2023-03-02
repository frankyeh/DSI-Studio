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

#include <filesystem>

void show_view(QGraphicsScene& scene,QImage I);

db_window::db_window(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc_) :
    QMainWindow(parent),vbc(vbc_),
    ui(new Ui::db_window)
{
    ui->setupUi(this);
    ui->report->setText(vbc->handle->db.report.c_str());
    ui->vbc_view->setScene(&vbc_scene);

    ui->x_pos->setMaximum(vbc->handle->dim[0]-1);
    ui->y_pos->setMaximum(vbc->handle->dim[1]-1);
    ui->z_pos->setMaximum(vbc->handle->dim[2]-1);

    connect(ui->slice_pos,SIGNAL(valueChanged(int)),this,SLOT(on_subject_list_itemSelectionChanged()));
    connect(ui->view_y,SIGNAL(toggled(bool)),this,SLOT(on_view_x_toggled(bool)));
    connect(ui->view_z,SIGNAL(toggled(bool)),this,SLOT(on_view_x_toggled(bool)));

    connect(ui->zoom,SIGNAL(valueChanged(double)),this,SLOT(on_subject_list_itemSelectionChanged()));
    connect(ui->add,SIGNAL(clicked()),this,SLOT(on_actionAdd_DB_triggered()));

    on_view_x_toggled(true);
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

    QMessageBox::StandardButton r = QMessageBox::question( this, "DSI Studio",
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
    ui->subject_list->setColumnCount(vbc->handle->db.feature_titles.size() + 1);
    ui->subject_list->setColumnWidth(0,100);
    ui->subject_list->setRowCount(vbc->handle->db.num_subjects);
    QStringList header;
    header << "id";
    for(auto& str : vbc->handle->db.feature_titles)
        header << str.c_str();
    ui->subject_list->setHorizontalHeaderLabels(header);

    for(unsigned int i = 0;i < vbc->handle->db.num_subjects;++i)
    {
        ui->subject_list->setItem(i,0, new QTableWidgetItem(QString(vbc->handle->db.subject_names[i].c_str())));
        for(unsigned int j = 0;j < vbc->handle->db.feature_location.size();++j)
            ui->subject_list->setItem(i,j+1, new QTableWidgetItem(QString(vbc->handle->db.items[i*vbc->handle->db.titles.size() + vbc->handle->db.feature_location[j]].c_str())));
    }

}

void db_window::on_subject_list_itemSelectionChanged()
{
    if(ui->subject_list->currentRow() == -1 ||
            ui->subject_list->currentRow() >= vbc->handle->db.subject_qa.size())
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
    tipl::normalize(slice,255.0f);
    tipl::color_image color_slice(slice.shape());
    std::copy(slice.begin(),slice.end(),color_slice.begin());

    QImage qimage((unsigned char*)&*color_slice.begin(),color_slice.width(),color_slice.height(),QImage::Format_RGB32);
    vbc_slice_image = qimage.scaled(color_slice.width()*ui->zoom->value(),color_slice.height()*ui->zoom->value());
    if(!ui->view_z->isChecked())
        vbc_slice_image = vbc_slice_image.mirrored();
    show_view(vbc_scene,vbc_slice_image);
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
    std::ofstream out(filename.toLocal8Bit().begin());
    for(unsigned int index = 0;index < vbc->handle->db.num_subjects;++index)
        out << vbc->handle->db.subject_names[index] << std::endl;
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
    std::ofstream out(filename.toLocal8Bit().begin());
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
    ui->report->setText(vbc->handle->db.report.c_str());
}


void db_window::on_delete_subject_clicked()
{
    if(ui->subject_list->currentRow() >= 0 && vbc->handle->db.num_subjects > 1)
    {
        unsigned int index = ui->subject_list->currentRow();
        vbc->handle->db.remove_subject(index);
        if(index < ui->subject_list->rowCount())
            ui->subject_list->removeRow(index);
    }
}

void db_window::on_actionCalculate_change_triggered()
{
    std::unique_ptr<match_db> mdb(new match_db(this,vbc));
    if(mdb->exec() == QDialog::Accepted)
        update_db();
}

void db_window::on_actionSave_DB_as_triggered()
{
    QString filename = QFileDialog::getSaveFileName(
                           this,
                           "Save Database",
                           windowTitle()+".modified.db.fib.gz",
                           "Database files (*db?fib.gz *fib.gz);;All files (*)");
    if (filename.isEmpty())
        return;
    progress prog_("saving ",std::filesystem::path(filename.toStdString()).filename().string().c_str());
    if(!vbc->handle->db.demo.empty() && !vbc->handle->db.parse_demo())
        QMessageBox::information(this,"DSI Studio",
        QString("demographics not saved due to mismatch: ") + vbc->handle->db.error_msg.c_str());

    if(vbc->handle->db.save_db(filename.toStdString().c_str()))
        QMessageBox::information(this,"DSI Studio","File saved");
    else
        QMessageBox::critical(this,"ERROR",vbc->handle->db.error_msg.c_str());
}

void db_window::on_move_down_clicked()
{

    if(ui->subject_list->currentRow() >= vbc->handle->db.num_subjects-1)
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
                           "Open Database files",
                           windowTitle(),
                           "Database files (*db?fib.gz *fib.gz);;All files (*)");
    if (filenames.isEmpty())
        return;
    for(int i =0;i < filenames.count();++i)
    {
        std::shared_ptr<fib_data> handle(new fib_data);
        progress prog_("adding data");
        if(!handle->load_from_file(filenames[i].toStdString().c_str()))
        {
            QMessageBox::critical(this,"ERROR",handle->error_msg.c_str());
            return;
        }
        if(!handle->is_mni)
        {
            QMessageBox::critical(this,"ERROR",filenames[i] + " is not from the QSDR reconstruction.");
            break;
        }

        if(handle->db.has_db())
        {
            if(!vbc->handle->db.add_db(handle->db))
            {
                QMessageBox::critical(this,"ERROR",vbc->handle->db.error_msg.c_str());
                break;
            }
            continue;
        }
        if(handle->has_odfs())
        {
            if(!vbc->handle->db.add_subject_file(filenames[i].toStdString(),QFileInfo(filenames[i]).baseName().toStdString()))
            {
                QMessageBox::information(this,"ERROR",vbc->handle->db.error_msg.c_str());
                break;
            }
        }
    }
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
    selected.resize(vbc->handle->db.num_subjects);
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
    tipl::image<3> I;
    vbc->handle->db.get_subject_volume(uint32_t(ui->subject_list->currentRow()),I);
    if(gz_nifti::save_to_file(filename.toStdString().c_str(),I,vbc->handle->vs,vbc->handle->trans_to_mni,true))
        QMessageBox::information(this,"File saved",filename);
    else
        QMessageBox::critical(this,"ERROR","Cannot save file.");
}

void db_window::on_actionAll_Subjects_triggered()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Output directory",
                                "");
    if(dir.isEmpty())
        return;
    progress prog_("exporting ",dir.toStdString().c_str());
    for(size_t i = 0;progress::at(i,vbc->handle->db.subject_names.size());++i)
    {
        QString file_name = dir + "\\"+
                vbc->handle->db.subject_names[i].c_str()+"."+
                vbc->handle->db.index_name.c_str()+".nii.gz";
        tipl::image<3> I;
        vbc->handle->db.get_subject_volume(uint32_t(i),I);
        gz_nifti out;
        out.set_voxel_size(vbc->handle->vs);
        out.set_image_transformation(vbc->handle->trans_to_mni);
        out << I;
        if(!out.save_to_file(file_name.toLocal8Bit().begin()))
        {
            QMessageBox::critical(this,"ERROR","Cannot save file.");
            return;
        }
    }
    QMessageBox::information(this,"DSI Studio","Files exported");
}

void db_window::on_actionOpen_Demographics_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
                this,
                "Open demographics",
                QFileInfo(vbc->handle->fib_file_name.c_str()).absoluteDir().absolutePath(),
                "Text or CSV file (*.txt *.csv);;All files (*)");
    if(filename.isEmpty())
        return;
    if(vbc->handle->db.parse_demo(filename.toStdString()))
    {
        QMessageBox::information(this,"DSI Studio","Demographics Loaded");
        update_subject_list();
    }
    else
        QMessageBox::critical(this,"ERROR",vbc->handle->db.error_msg.c_str());
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
                "Text or CSV file (*.txt *.csv);;All files (*)");
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
        QMessageBox::information(this,"DSI Studio","File Saved");
    else
        QMessageBox::critical(this,"ERROR",vbc->handle->db.error_msg.c_str());

}


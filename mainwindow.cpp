#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "image/image.hpp"
#include <QFileDialog>
#include <QDateTime>
#include <QUrl>
#include <QMessageBox>
#include <QProgressDialog>
#include <QDragEnterEvent>
#include <QMimeData>
#include <regtoolbox.h>
#include <qmessagebox.h>
#include "filebrowser.h"
#include "reconstruction/reconstruction_window.h"
#include "dsi_interface_static_link.h"
#include "prog_interface_static_link.h"
#include "tracking/tracking_window.h"
#include "mainwindow.h"
#include "dicom/dicom_parser.h"
#include "ui_mainwindow.h"
#include "simulation.h"
#include "view_image.h"
#include "mapping/atlas.hpp"
#include "libs/gzip_interface.hpp"
#include "vbc/vbc_database.h"
#include "libs/tracking/fib_data.hpp"
#include "manual_alignment.h"
#include "connectometry/individual_connectometry.hpp"
#include "connectometry/createdbdialog.h"
#include "connectometry/db_window.h"
#include "connectometry/group_connectometry.hpp"
#include "program_option.hpp"
#include "libs/dsi/image_model.hpp"
extern program_option po;
int rec(void);
int trk(void);
int src(void);
int ana(void);
int exp(void);
int atl(void);
int cnt(void);
int vis(void);
int ren(void);
extern std::vector<atlas> atlas_list;
extern std::auto_ptr<QProgressDialog> progressDialog;

MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent),
        ui(new Ui::MainWindow)
{
    setAcceptDrops(true);


    progressDialog.reset(new QProgressDialog);
    progressDialog->hide();
    ui->setupUi(this);
    ui->recentFib->setColumnCount(3);
    ui->recentFib->setColumnWidth(0,300);
    ui->recentFib->setColumnWidth(1,300);
    ui->recentFib->setColumnWidth(2,200);
    ui->recentFib->setAlternatingRowColors(true);
    ui->recentSrc->setColumnCount(3);
    ui->recentSrc->setColumnWidth(0,300);
    ui->recentSrc->setColumnWidth(1,300);
    ui->recentSrc->setColumnWidth(2,200);
    ui->recentSrc->setAlternatingRowColors(true);
    QObject::connect(ui->recentFib,SIGNAL(cellDoubleClicked(int,int)),this,SLOT(open_fib_at(int,int)));
    QObject::connect(ui->recentSrc,SIGNAL(cellDoubleClicked(int,int)),this,SLOT(open_src_at(int,int)));
    updateRecentList();

    if (settings.contains("WORK_PATH"))
        ui->workDir->addItems(settings.value("WORK_PATH").toStringList());
    else
        ui->workDir->addItem(QDir::currentPath());

    ui->toolBox->setCurrentIndex(0);

    QSettings settings;
    restoreGeometry(settings.value("main_geometry").toByteArray());
    restoreState(settings.value("main_state").toByteArray());
}
void MainWindow::dragEnterEvent(QDragEnterEvent *event)
{
    if(event->mimeData()->hasUrls())
    {
        event->acceptProposedAction();
    }
}

void MainWindow::dropEvent(QDropEvent *event)
{
    event->acceptProposedAction();
    QList<QUrl> droppedUrls = event->mimeData()->urls();
    int droppedUrlCnt = droppedUrls.size();
    QStringList files;
    for(int i = 0; i < droppedUrlCnt; i++)
        files << droppedUrls[i].toLocalFile();

    if(files.size() == 1)
    {
        if(QFileInfo(files[0]).completeSuffix() == "fib.gz")
        {
            loadFib(files[0]);
            return;
        }
        if(QFileInfo(files[0]).completeSuffix() == "src.gz")
        {
            loadSrc(files);
            return;
        }
    }

    dicom_parser* dp = new dicom_parser(files,this);
    dp->setAttribute(Qt::WA_DeleteOnClose);
    dp->showNormal();

}

void MainWindow::open_fib_at(int row,int)
{
    loadFib(ui->recentFib->item(row,1)->text() + "/" +
            ui->recentFib->item(row,0)->text());
}

void MainWindow::open_src_at(int row,int)
{
    loadSrc(QStringList() << (ui->recentSrc->item(row,1)->text() + "/" +
            ui->recentSrc->item(row,0)->text()));
}


MainWindow::~MainWindow()
{
    QStringList workdir_list;
    for (int index = 0;index < 10 && index < ui->workDir->count();++index)
        workdir_list << ui->workDir->itemText(index);
    std::swap(workdir_list[0],workdir_list[ui->workDir->currentIndex()]);
    settings.setValue("WORK_PATH", workdir_list);
    settings.setValue("main_geometry", saveGeometry());
    settings.setValue("main_state", saveState());
    delete ui;

}

\
void MainWindow::updateRecentList(void)
{
    {
        QStringList file_list = settings.value("recentFibFileList").toStringList();
        for (int index = 0;index < file_list.size();)
            if(!QFileInfo(file_list[index]).exists())
                file_list.removeAt(index);
            else
                ++index;
            ui->recentFib->clear();
            ui->recentFib->setRowCount(file_list.size());
        for (int index = 0;index < file_list.size();++index)
        {
            ui->recentFib->setRowHeight(index,20);
            ui->recentFib->setItem(index, 0, new QTableWidgetItem(QFileInfo(file_list[index]).fileName()));
            ui->recentFib->setItem(index, 1, new QTableWidgetItem(QFileInfo(file_list[index]).absolutePath()));
            ui->recentFib->setItem(index, 2, new QTableWidgetItem(QFileInfo(file_list[index]).created().toString()));
            ui->recentFib->item(index,0)->setFlags(ui->recentFib->item(index,0)->flags() & ~Qt::ItemIsEditable);
            ui->recentFib->item(index,1)->setFlags(ui->recentFib->item(index,1)->flags() & ~Qt::ItemIsEditable);
            ui->recentFib->item(index,2)->setFlags(ui->recentFib->item(index,2)->flags() & ~Qt::ItemIsEditable);
        }
    }
    {
        QStringList file_list = settings.value("recentSrcFileList").toStringList();
        for (int index = 0;index < file_list.size();)
            if(!QFileInfo(file_list[index]).exists())
                file_list.removeAt(index);
            else
                ++index;
        ui->recentSrc->clear();
        ui->recentSrc->setRowCount(file_list.size());
        for (int index = 0;index < file_list.size();++index)
        {
            ui->recentSrc->setRowHeight(index,20);
            ui->recentSrc->setItem(index, 0, new QTableWidgetItem(QFileInfo(file_list[index]).fileName()));
            ui->recentSrc->setItem(index, 1, new QTableWidgetItem(QFileInfo(file_list[index]).absolutePath()));
            ui->recentSrc->setItem(index, 2, new QTableWidgetItem(QFileInfo(file_list[index]).created().toString()));
            ui->recentSrc->item(index,0)->setFlags(ui->recentSrc->item(index,0)->flags() & ~Qt::ItemIsEditable);
            ui->recentSrc->item(index,1)->setFlags(ui->recentSrc->item(index,1)->flags() & ~Qt::ItemIsEditable);
            ui->recentSrc->item(index,2)->setFlags(ui->recentSrc->item(index,2)->flags() & ~Qt::ItemIsEditable);
        }
    }
    QStringList header;
    header << "File Name" << "Directory" << "Date";
    ui->recentFib->setHorizontalHeaderLabels(header);
    ui->recentSrc->setHorizontalHeaderLabels(header);
}

void MainWindow::addFib(QString filename)
{
    // update recent file list
    QStringList files = settings.value("recentFibFileList").toStringList();
    files.removeAll(filename);
    files.prepend(filename);
    while (files.size() > MaxRecentFiles)
        files.removeLast();
    settings.setValue("recentFibFileList", files);
    updateRecentList();
}

void MainWindow::addSrc(QString filename)
{
    // update recent file list
    QStringList files = settings.value("recentSrcFileList").toStringList();
    files.removeAll(filename);
    files.prepend(filename);
    while (files.size() > MaxRecentFiles)
        files.removeLast();
    settings.setValue("recentSrcFileList", files);
    updateRecentList();
}

void MainWindow::loadFib(QString filename)
{
    std::string file_name = filename.toLocal8Bit().begin();
    begin_prog("load fib");
    std::shared_ptr<fib_data> new_handle(new fib_data);
    if (!new_handle->load_from_file(&*file_name.begin()))
    {
        if(!prog_aborted())
            QMessageBox::information(this,"error",new_handle->error_msg.c_str(),0);
        return;
    }
    tracking_window* new_mdi = new tracking_window(this,new_handle);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->setWindowTitle(filename);
    new_mdi->showNormal();
    addFib(filename);
    add_work_dir(QFileInfo(filename).absolutePath());
    QDir::setCurrent(QFileInfo(filename).absolutePath());
}

void MainWindow::loadSrc(QStringList filenames)
{
    if(filenames.empty())
    {
        QMessageBox::information(this,"Error","Cannot find SRC.gz files in the directory. Please create SRC files first.");
        return;
    }
    try
    {

        reconstruction_window* new_mdi = new reconstruction_window(filenames,this);
        new_mdi->setAttribute(Qt::WA_DeleteOnClose);
        new_mdi->show();
        QDir::setCurrent(QFileInfo(filenames[0]).absolutePath());
        if(filenames.size() == 1)
        {
            addSrc(filenames[0]);
            add_work_dir(QFileInfo(filenames[0]).absolutePath());
        }
    }
    catch(...)
    {

    }

}


void MainWindow::openRecentFibFile(void)
{
    QAction *action = qobject_cast<QAction *>(sender());
    loadFib(action->data().toString());
}
void MainWindow::openRecentSrcFile(void)
{
    QAction *action = qobject_cast<QAction *>(sender());
    loadSrc(QStringList() << action->data().toString());
}

void MainWindow::on_OpenDICOM_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                this,
                                "Open Images files",
                                ui->workDir->currentText(),
                                "Image files (*.dcm *.hdr *.nii *nii.gz *.fdf 2dseq subject);;All files (*)" );
    if ( filenames.isEmpty() )
        return;

    add_work_dir(QFileInfo(filenames[0]).absolutePath());
    if(QFileInfo(filenames[0]).completeBaseName() == "subject")
    {
        image::io::bruker_info subject_file;
        if(!subject_file.load_from_file(filenames[0].toLocal8Bit().begin()))
            return;
        QString dir = QFileInfo(filenames[0]).absolutePath();
        filenames.clear();
        for(unsigned int i = 1;i < 100;++i)
        if(QDir(dir + "/" +QString::number(i)).exists())
        {
            bool is_dwi =false;
            // has dif info in the method file
            {
                image::io::bruker_info method_file;
                QString method_name = dir + "/" +QString::number(i)+"/method";
                if(method_file.load_from_file(method_name.toLocal8Bit().begin()) &&
                   method_file["PVM_DwEffBval"].length())
                    is_dwi = true;
            }
            // has dif info in the imnd file
            {
                image::io::bruker_info imnd_file;
                QString imnd_name = dir + "/" +QString::number(i)+"/imnd";
                if(imnd_file.load_from_file(imnd_name.toLocal8Bit().begin()) &&
                   imnd_file["IMND_diff_b_value"].length())
                    is_dwi = true;
            }
            if(is_dwi)
                filenames.push_back(dir + "/" +QString::number(i)+"/pdata/1/2dseq");
        }
        if(filenames.size() == 0)
        {
            QMessageBox::information(this,"Error","No diffusion data in this subject",0);
            return;
        }
        std::string file_name(subject_file["SUBJECT_study_name"]);
        file_name.erase(std::remove(file_name.begin(),file_name.end(),' '),file_name.end());
        dicom_parser* dp = new dicom_parser(filenames,this);
        dp->set_name(dir + "/" + file_name.c_str() + ".src.gz");
        dp->setAttribute(Qt::WA_DeleteOnClose);
        dp->showNormal();
        return;
    }

    if(QFileInfo(filenames[0]).completeSuffix() == "dcm")
    {
        QString sel = QString("*.")+QFileInfo(filenames[0]).suffix();
        QDir directory = QFileInfo(filenames[0]).absoluteDir();
        QStringList file_list = directory.entryList(QStringList(sel),QDir::Files|QDir::NoSymLinks);
        if(file_list.size() > filenames.size())
        {
            QString msg =
              QString("There are %1 %2 files in the directory. Select all?").arg(file_list.size()).arg(QFileInfo(filenames[0]).suffix());
            int result = QMessageBox::information(this,"Input images",msg,
                                     QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel);
            if(result == QMessageBox::Cancel)
                return;
            if(result == QMessageBox::Yes)
            {
                filenames = file_list;
                for(int index = 0;index < filenames.size();++index)
                    filenames[index] = directory.absolutePath() + "/" + filenames[index];
            }
        }
    }
    dicom_parser* dp = new dicom_parser(filenames,this);
    dp->setAttribute(Qt::WA_DeleteOnClose);
    dp->showNormal();
}

void MainWindow::on_Reconstruction_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                           this,
                           "Open Src files",
                           ui->workDir->currentText(),
                           "Src files (*src.gz *.src);;All files (*)" );
    if (filenames.isEmpty())
        return;
    loadSrc(filenames);
}

void MainWindow::on_FiberTracking_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,
                           "Open Fib files",
                           ui->workDir->currentText(),
                           "Fib files (*fib.gz *.fib *nii.gz *.nii 2dseq);;All files (*)");
    if (filename.isEmpty())
        return;
    image::basic_image<float,3> I;
    float vs[3];
    if(QFileInfo(filename).completeSuffix() == "nii" ||
            QFileInfo(filename).completeSuffix() == "nii.gz")
    {
        gz_nifti header;
        if(!header.load_from_file(filename.toLocal8Bit().begin()))
        {
            QMessageBox::information(this,"Error","Invalid NIFTI format",0);
            return;
        }
        header.toLPS(I);
        header.get_voxel_size(vs);
    }
    if(QFileInfo(filename).fileName() == "2dseq")
    {
        image::io::bruker_2dseq bruker_header;
        if(!bruker_header.load_from_file(filename.toLocal8Bit().begin()))
        {
            QMessageBox::information(this,"DSI Studio","Invalid 2dseq format",0);
            return;
        }
        bruker_header.get_image().swap(I);
        bruker_header.get_voxel_size(vs);
    }
    if(!I.empty())
    {
        std::shared_ptr<fib_data> new_handle(new fib_data);
        new_handle->mat_reader.add("dimension",I.geometry().begin(),3,1);
        new_handle->mat_reader.add("voxel_size",vs,3,1);
        new_handle->mat_reader.add("image",&*I.begin(),I.size(),1);
        new_handle->load_from_mat();
        new_handle->dir.index_name[0] = "image";
        new_handle->view_item[0].name = "image";
        tracking_window* new_mdi = new tracking_window(this,new_handle);
        new_mdi->setAttribute(Qt::WA_DeleteOnClose);
        new_mdi->setWindowTitle(filename);
        new_mdi->showNormal();
        new_mdi->set_data("roi_fiber",0);
        new_mdi->scene.show_slice();
    }
    else
        loadFib(filename);
}

void check_name(std::string& name)
{
    for(unsigned int index = 0;index < name.size();++index)
        if((name[index] < '0' || name[index] > '9') &&
           (name[index] < 'a' || name[index] > 'z') &&
           (name[index] < 'A' || name[index] > 'Z') &&
                name[index] != '.')
            name[index] = '_';
}

bool RenameDICOMToDir(QString FileName, QString ToDir)
{
    std::string person, sequence, imagename;
    {
        image::io::dicom header;
        if (!header.load_from_file(FileName.toLocal8Bit().begin()))
            return false;

        header.get_patient(person);
        header.get_sequence(sequence);
        header.get_image_name(imagename);
    }
    check_name(person);
    check_name(sequence);
    check_name(imagename);

    QString Person(person.c_str()), Sequence(sequence.c_str()),
    ImageName(imagename.c_str());

    ToDir += "/";
    ToDir += Person;
    if (!QDir(ToDir).exists())
    {
        if(!QDir(ToDir).mkdir(ToDir))
        {
            std::cout << "Cannot create dir " << ToDir.toStdString() << std::endl;
            return false;
        }
    }


    ToDir += "/";
    ToDir += Sequence;
    if (!QDir(ToDir).exists())
    {
        if(!QDir(ToDir).mkdir(ToDir))
        {
            std::cout << "Cannot create dir " << ToDir.toStdString() << std::endl;
            return false;
        }
    }

    ToDir += "/";
    ToDir += ImageName;
    std::cout << FileName.toStdString() << "->" << ToDir.toStdString() << std::endl;
    return QFile(FileName).rename(FileName,ToDir);
}



void MainWindow::on_RenameDICOM_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                                this,
                                "Open DICOM files",
                                ui->workDir->currentText(),
                                "All files (*)" );
    if ( filenames.isEmpty() )
        return;
    begin_prog("Rename DICOM Files");
    for (unsigned int index = 0;check_prog(index,filenames.size());++index)
        RenameDICOMToDir(filenames[index],QFileInfo(filenames[index]).absolutePath());
}


void MainWindow::add_work_dir(QString dir)
{
    if(ui->workDir->findText(dir) != -1)
        ui->workDir->removeItem(ui->workDir->findText(dir));
    ui->workDir->insertItem(0,dir);
    ui->workDir->setCurrentIndex(0);
}


void MainWindow::on_browseDir_clicked()
{
    QString filename =
        QFileDialog::getExistingDirectory(this,"Browse Directory",
                                          ui->workDir->currentText());
    if ( filename.isEmpty() )
        return;
    add_work_dir(filename);
}

void MainWindow::on_simulateMRI_clicked()
{
    (new Simulation(this,ui->workDir->currentText()))->show();
}

QStringList GetSubDir(QString Dir)
{
    QStringList sub_dirs;
    QStringList dirs = QDir(Dir).entryList(QStringList("*"),
                                            QDir::Dirs | QDir::NoSymLinks | QDir::NoDotAndDotDot);
    sub_dirs << Dir;
    for(int index = 0;index < dirs.size();++index)
    {
        QString new_dir = Dir + "/" + dirs[index];
        sub_dirs << GetSubDir(new_dir);
    }
    return sub_dirs;
}
void MainWindow::on_RenameDICOMDir_clicked()
{
    QString path =
        QFileDialog::getExistingDirectory(this,"Browse Directory",
                                          ui->workDir->currentText());
    if ( path.isEmpty() )
        return;
    QStringList dirs = GetSubDir(path);
    for(unsigned int index = 0;check_prog(index,dirs.size());++index)
    {
        QStringList files = QDir(dirs[index]).entryList(QStringList("*"),
                                    QDir::Files | QDir::NoSymLinks);
        set_title(QFileInfo(dirs[index]).fileName().toLocal8Bit().begin());
        for(unsigned int j = 0;j < files.size() && check_prog(index,dirs.size());++j)
        {
            set_title(files[j].toLocal8Bit().begin());
            RenameDICOMToDir(dirs[index] + "/" + files[j],path);
        }
    }
}

void MainWindow::on_vbc_clicked()
{
    CreateDBDialog* new_mdi = new CreateDBDialog(this,true);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->show();
}

void MainWindow::on_averagefib_clicked()
{
    CreateDBDialog* new_mdi = new CreateDBDialog(this,false);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->show();
}


/*
#include "mapping/mni_norm.hpp"
void MainWindow::on_warpImage_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                                this,
                                "Open tr.mat file",
                                ui->workDir->currentText(),
                                "MAT files (*.mat)" );
    if (filename.isEmpty())
        return;
    MNINorm mni;
    if(!mni.load_transformation_matrix(filename.toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"Error","Invalid tr.mat format",0);
        return;
    }

    filename = QFileDialog::getOpenFileName(
                                this,
                                "Open image file",
                                ui->workDir->currentText(),
                                "Nifti files (*.nii)" );
    if (filename.isEmpty())
        return;

    image::basic_image<float,3> I,out;
    {
        image::io::nifti read;
        if(!read.load_from_file(filename.toLocal8Bit().begin()))
        {
            QMessageBox::information(this,"Error","Cannot open image file",0);
            return;
        }
        read >> I;
        if(read.nif_header.srow_x[0] < 0)
            image::flip_y(I);
        else
            image::flip_xy(I);
    }
    mni.warp(I,out);
    {
        image::io::nifti write;
        image::flip_xy(out);
        write << out;
        float voxel_size[3]={1,1,1};
        write.set_voxel_size(voxel_size);
        float ir[12] = {1.0,0.0,0.0,-78.0,
                        0.0,1.0,0.0,-112.0,
                        0.0,0.0,1.0,-50.0};
        write.set_image_transformation(ir);
        write.save_to_file((filename+".warp.nii").toLocal8Bit().begin());
    }
}
*/
bool load_all_files(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files);
bool find_bval_bvec(const char* file_name,QString& bval,QString& bvec);
bool load_4d_nii(const char* file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files);
QString get_src_name(QString file_name);

void MainWindow::on_batch_src_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                ui->workDir->currentText());
    if(dir.isEmpty())
        return;
    add_work_dir(dir);

    {
        QStringList dir_list;
        dir_list << dir;
        bool all = false;
        bool choice = false;
        begin_prog("batch creating src");
        for(unsigned int i = 0;check_prog(i,dir_list.size()) && !prog_aborted();++i)
        {
            QDir cur_dir = dir_list[i];
            QStringList new_list = cur_dir.entryList(QStringList(""),QDir::AllDirs|QDir::NoDotAndDotDot);
            for(unsigned int index = 0;index < new_list.size();++index)
                dir_list << cur_dir.absolutePath() + "/" + new_list[index];


            std::vector<std::shared_ptr<DwiHeader> > dwi_files;


            if(QFileInfo(dir_list[i] + "/data.nii.gz").exists() &&
               QFileInfo(dir_list[i] + "/bvals").exists() &&
               QFileInfo(dir_list[i] + "/bvecs").exists() &&
               load_4d_nii(QString(dir_list[i] + "/data.nii.gz").toLocal8Bit().begin(),dwi_files))
            {
                if(!DwiHeader::has_b_table(dwi_files))
                {
                    std::ofstream(QString(dir_list[i] + "/data.nii.gz.b_table_mismatch.txt").toLocal8Bit().begin());
                    continue;
                }
                DwiHeader::output_src(QString(dir_list[i] + "/data.src.gz").toLocal8Bit().begin(),dwi_files,0);
                continue;
            }

            QStringList nifti_file_list = cur_dir.entryList(QStringList("*.nii.gz") << "*.nii",QDir::Files|QDir::NoSymLinks);
            for (unsigned int index = 0;index < nifti_file_list.size();++index)
            {
                QString bval,bvec;
                if(find_bval_bvec(QString(dir_list[i] + "/" + nifti_file_list[index]).toLocal8Bit().begin(),bval,bvec))
                {
                    if(!load_4d_nii(QString(dir_list[i] + "/" + nifti_file_list[index]).toLocal8Bit().begin(),dwi_files))
                    {
                        std::ofstream(QString(dir_list[i] + "/" + nifti_file_list[index] + ".invalid_format.txt").toLocal8Bit().begin());
                        continue;
                    }
                    if(!DwiHeader::has_b_table(dwi_files))
                    {
                        std::ofstream(QString(dir_list[i] + "/" + nifti_file_list[index] + ".b_table_mismatch.txt").toLocal8Bit().begin());
                        continue;
                    }
                    DwiHeader::output_src(QString(dir_list[i] + "/" +
                        QFileInfo(nifti_file_list[index]).baseName() + ".src.gz").toLocal8Bit().begin(),dwi_files,0);
                    continue;
                }
            }

            QStringList dicom_file_list = cur_dir.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
            if(dicom_file_list.empty())
                continue;
            for (unsigned int index = 0;index < dicom_file_list.size();++index)
                dicom_file_list[index] = dir_list[i] + "/" + dicom_file_list[index];
            QString output = dir_list[i] + "/" + QFileInfo(get_src_name(dicom_file_list[0])).baseName()+".src.gz";
            if(QFileInfo(output).exists())
            {
                if(!all)
                {
                    QMessageBox msgBox;
                    msgBox.setText(QString("Existing SRC file ") + output);
                    msgBox.setInformativeText("Overwrite?");
                    msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::YesToAll | QMessageBox::No | QMessageBox::NoAll);
                    msgBox.setDefaultButton(QMessageBox::Save);
                    switch(msgBox.exec())
                    {
                    case QMessageBox::YesToAll:
                        all = true;
                    case QMessageBox::Yes:
                        choice = true;
                        break;
                    case QMessageBox::NoAll:
                        all = true;
                    case QMessageBox::No:
                        choice = false;
                        break;
                    }
                }
                if(!choice)
                    continue;
            }
            if(!load_all_files(dicom_file_list,dwi_files) || prog_aborted())
                continue;
            if(dwi_files.size() == 1) //MPRAGE or T2W
            {
                std::sort(dicom_file_list.begin(),dicom_file_list.end(),compare_qstring());
                image::io::volume v;
                image::io::dicom header;
                std::vector<std::string> file_list;
                for(unsigned int index = 0;index < dicom_file_list.size();++index)
                    file_list.push_back(dicom_file_list[index].toLocal8Bit().begin());
                if(!v.load_from_files(file_list,file_list.size()) ||
                   !header.load_from_file(dicom_file_list[0].toLocal8Bit().begin()))
                    continue;
                image::basic_image<float,3> I;
                image::vector<3> vs;
                v >> I;
                v.get_voxel_size(vs.begin());
                gz_nifti nii_out;
                image::flip_xy(I);
                nii_out << I;
                nii_out.set_voxel_size(vs);


                std::string manu,make,seq,report;
                header.get_text(0x0008,0x0070,manu);//Manufacturer
                header.get_text(0x0008,0x1090,make);
                header.get_text(0x0018,0x1030,seq);
                std::replace(manu.begin(),manu.end(),' ',(char)0);
                make.erase(std::remove(make.begin(),make.end(),' '),make.end());
                std::ostringstream out;
                out << manu.c_str() << " " << make.c_str() << " " << seq
                    << ".TE=" << header.get_float(0x0018,0x0081) << ".TR=" << header.get_float(0x0018,0x0080)  << ".";
                report = out.str();
                if(report.size() < 80)
                    report.resize(80);
                nii_out.set_descrip(report.c_str());
                QString output_name = QFileInfo(get_src_name(dicom_file_list[0])).absolutePath() + "/" +
                                      QFileInfo(get_src_name(dicom_file_list[0])).baseName()+ "."+seq.c_str() + ".nii.gz";
                nii_out.save_to_file(output_name.toLocal8Bit().begin());
            }
            else
            {
                for(unsigned int index = 0;index < dwi_files.size();++index)
                    if(dwi_files[index]->get_bvalue() < 100)
                    {
                        dwi_files[index]->set_bvalue(0);
                        dwi_files[index]->set_bvec(0,0,0);
                    }
                DwiHeader::output_src(output.toLocal8Bit().begin(),dwi_files,0);
            }
        }
    }
}

QStringList search_files(QString dir,QString filter);
void MainWindow::on_batch_reconstruction_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                ui->workDir->currentText());
    if(dir.isEmpty())
        return;

    loadSrc(search_files(dir,"*.src.gz"));
}

void MainWindow::on_view_image_clicked()
{
    QStringList filename = QFileDialog::getOpenFileNames(
                                this,
                                "Open Image",
                                ui->workDir->currentText(),
                                "image files (*.nii *nii.gz *.dcm 2dseq *fib.gz *src.gz)" );
    if(filename.isEmpty())
        return;
    view_image* dialog = new view_image(this);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    if(!dialog->open(filename))
    {
        delete dialog;
        return;
    }
    dialog->show();

}

void MainWindow::on_workDir_currentTextChanged(const QString &arg1)
{
    QDir::setCurrent(arg1);
}

void MainWindow::on_bruker_browser_clicked()
{
    FileBrowser* bw = new FileBrowser(this);
    bw->setAttribute(Qt::WA_DeleteOnClose);
    bw->showNormal();
}

void MainWindow::on_individual_connectometry_clicked()
{
    individual_connectometry* indi = new individual_connectometry(this);
        indi->setAttribute(Qt::WA_DeleteOnClose);
        indi->showNormal();
}


bool MainWindow::load_db(std::shared_ptr<vbc_database>& database,QString& filename)
{
    filename = QFileDialog::getOpenFileName(
                           this,
                           "Open Database files",
                           ui->workDir->currentText(),
                           "Database files (*db?fib.gz);;All files (*)");
    if (filename.isEmpty())
        return false;
    QDir::setCurrent(QFileInfo(filename).absolutePath());
    add_work_dir(QFileInfo(filename).absolutePath());
    database = std::make_shared<vbc_database>();
    begin_prog("reading connectometry db");
    if(!database->load_database(filename.toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"Error",database->error_msg.c_str(),0);
        return false;
    }
    return true;
}

void MainWindow::on_open_db_clicked()
{
    QString filename;
    std::shared_ptr<vbc_database> database;
    if(!load_db(database,filename))
        return;
    db_window* db = new db_window(this,database);
    db->setWindowTitle(filename);
    db->setAttribute(Qt::WA_DeleteOnClose);
    db->show();
}

void MainWindow::on_group_connectometry_clicked()
{
    QString filename;
    std::shared_ptr<vbc_database> database;
    if(!load_db(database,filename))
        return;
    group_connectometry* group_cnt = new group_connectometry(this,database,filename,true);
    group_cnt->setAttribute(Qt::WA_DeleteOnClose);
    group_cnt->show();
}


void MainWindow::on_run_cmd_clicked()
{
    po.init(ui->cmd_line->text().toStdString());
    if (!po.has("action") || !po.has("source"))
    {
        std::cout << "invalid command, use --help for more detail" << std::endl;
        return;
    }
    QDir::setCurrent(QFileInfo(po.get("source").c_str()).absolutePath());
    if(po.get("action") == std::string("rec"))
        rec();
    if(po.get("action") == std::string("trk"))
        trk();
    if(po.get("action") == std::string("src"))
        src();
    if(po.get("action") == std::string("ana"))
        ana();
    if(po.get("action") == std::string("exp"))
        exp();
    if(po.get("action") == std::string("atl"))
        atl();
    if(po.get("action") == std::string("cnt"))
        cnt();
    if(po.get("action") == std::string("vis"))
        vis();
    if(po.get("action") == std::string("ren"))
        ren();
}


void calculate_shell(const std::vector<float>& bvalues,std::vector<unsigned int>& shell);
bool is_dsi_half_sphere(const std::vector<unsigned int>& shell);
bool need_scheme_balance(const std::vector<unsigned int>& shell);
void MainWindow::on_ReconstructSRC_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                ui->workDir->currentText());
    if(dir.isEmpty())
        return;

    QStringList list = search_files(dir,"*.src.gz");

    for(int i = 0;i < list.size();++i)
    {
        std::shared_ptr<ImageModel> handle(std::make_shared<ImageModel>());
        if (!handle->load_from_file(list[i].toLocal8Bit().begin()))
        {
            QMessageBox::information(this,"error",QString("Cannot open ") +
                list[i] + " : " +handle->error_msg.c_str(),0);
            return;
        }

        float params[5] = {1.25,0,0,0,0};
        // determine the spatial resolution
        if(handle->voxel.vs[0] < 2.0)
            params[1] = 1; // 1 mm
        else
            params[1] = 2; // 2 mm

        handle->voxel.ti.init(8); // odf order of 8
        handle->voxel.odf_deconvolusion = 0;//ui->odf_sharpening->currentIndex() == 1 ? 1 : 0;
        handle->voxel.odf_decomposition = 0;//ui->odf_sharpening->currentIndex() == 2 ? 1 : 0;
        handle->voxel.odf_xyz[0] = 0;
        handle->voxel.odf_xyz[1] = 0;
        handle->voxel.odf_xyz[2] = 0;
        handle->voxel.csf_calibration = 0;
        handle->voxel.max_fiber_number = 5;
        handle->voxel.r2_weighted = 0;
        handle->voxel.reg_method = 3; // CDM
        handle->voxel.need_odf = 1; // output ODF
        handle->voxel.output_jacobian = 0;
        handle->voxel.output_mapping = 0;
        handle->voxel.output_diffusivity = 0;
        handle->voxel.output_tensor = 0;
        handle->voxel.output_rdi = 1;
        handle->voxel.thread_count = std::thread::hardware_concurrency();
        //checking half shell
        {
            std::vector<unsigned int> shell;
            calculate_shell(handle->voxel.bvalues,shell);
            handle->voxel.half_sphere = is_dsi_half_sphere(shell);
            handle->voxel.scheme_balance = need_scheme_balance(shell);
        }

        const char* msg = (const char*)reconstruction(handle.get(), 7 /*QSDR*/,
                                                      params,true /*check b-table*/);
        if (QFileInfo(msg).exists())
            continue;
        QMessageBox::information(this,"error",msg,0);
            return;
    }
}


void MainWindow::on_set_dir_clicked()
{
    QString dir =
        QFileDialog::getExistingDirectory(this,"Browse Directory","");
    if ( dir.isEmpty() )
        return;
    QDir::setCurrent(dir);
}

bool load_image_from_files(QStringList filenames,image::basic_image<float,3>& ref,image::vector<3>& vs);

void MainWindow::on_linear_reg_clicked()
{
    QStringList filename1 = QFileDialog::getOpenFileNames(
            this,"Open Subject Image",ui->workDir->currentText(),
            "Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if(filename1.isEmpty())
        return;


    QStringList filename2 = QFileDialog::getOpenFileNames(
            this,"Open Template Image",QFileInfo(filename1[0]).absolutePath(),
            "Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if(filename2.isEmpty())
        return;


    image::basic_image<float,3> ref1,ref2;
    image::vector<3> vs1,vs2;

    if(!load_image_from_files(filename1,ref1,vs1) ||
       !load_image_from_files(filename2,ref2,vs2))
        return;

    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,ref1,vs1,ref2,vs2,image::reg::affine,image::reg::mutual_info));
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;
}

void MainWindow::on_nonlinear_reg_clicked()
{
    RegToolBox* rt = new RegToolBox(this);
    rt->setAttribute(Qt::WA_DeleteOnClose);
    rt->showNormal();
}

std::string quality_check_src_files(QString dir);
void show_info_dialog(const std::string& title,const std::string& result);
void MainWindow::on_SRC_qc_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                ui->workDir->currentText());
    if(dir.isEmpty())
        return;
    begin_prog("checking SRC files");
    show_info_dialog("SRC report",quality_check_src_files(dir));
}

void MainWindow::on_parse_network_measures_clicked()
{
    QStringList filename = QFileDialog::getOpenFileNames(
            this,"Open Network Measures",ui->workDir->currentText(),
            "Text files (*.txt);;All files (*)" );
    if(filename.isEmpty())
        return;
    std::ofstream out((filename[0]+".collected.txt").toStdString().c_str());
    out << "Field\t";
    for(int i = 0;i < filename.size();++i)
        out << QFileInfo(filename[i]).baseName().toStdString() << "\t";
    out << std::endl;

    std::vector<std::string> line_output;
    for(int i = 0;i < filename.size();++i)
    {
        std::ifstream in(filename[i].toStdString().c_str());
        // global measures
        int line_index = 0;
        for(int j = 0;j < 19;++j,++line_index)
        {
            std::string t1,t2;
            in >> t1 >> t2;
            if(i == 0)
            {
                line_output.push_back(t1);
                line_output.back() += "\t";
            }
            line_output[line_index] += t2;
            line_output[line_index] += "\t";
        }
        std::vector<std::string> node_list;
        {
            std::string nodes;
            while(nodes.empty())
                in >> nodes; // skip the network measure header
            std::getline(in,nodes);
            std::istringstream nodestream(nodes);
            std::copy(std::istream_iterator<std::string>(nodestream),
                      std::istream_iterator<std::string>(),std::back_inserter(node_list));
        }
        // nodal measures
        for(int j = 0;j < 14;++j)
        {
            std::string line;
            std::getline(in,line);
            std::istringstream in2(line);
            std::string t1;
            in2 >> t1;
            for(int k = 0;k < node_list.size();++k,++line_index)
            {
                std::string t2;
                in2 >> t2;
                if(i==0)
                {
                    line_output.push_back(t1);
                    line_output.back() += "_";
                    line_output.back() += node_list[k];
                    line_output.back() += "\t";
                }
                line_output[line_index] += t2;
                line_output[line_index] += "\t";
            }
        }
    }
    for(int i = 0;i < line_output.size();++i)
        out << line_output[i] << std::endl;

    QMessageBox::information(this,"DSI Studio",QString("File saved to")+filename[0]+".collected.txt",0);

}

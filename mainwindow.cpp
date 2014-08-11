#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "image/image.hpp"
#include <QFileDialog>
#include <QDateTime>
#include <QUrl>
#include <QMessageBox>
#include <QProgressDialog>
#include <QDragEnterEvent>
#include <qmessagebox.h>
#include "reconstruction/reconstruction_window.h"
#include "dsi_interface_static_link.h"
#include "prog_interface_static_link.h"
#include "tracking_static_link.h"
#include "tracking/tracking_window.h"
#include "mainwindow.h"
#include "dicom/dicom_parser.h"
#include "ui_mainwindow.h"
#include "simulation.h"
#include "reconstruction/vbcdialog.h"
#include "view_image.h"
#include "mapping/atlas.hpp"
#include "libs/gzip_interface.hpp"
#include "tracking/vbc_dialog.hpp"
#include "vbc/vbc_database.h"

extern std::vector<atlas> atlas_list;
extern std::auto_ptr<QProgressDialog> progressDialog;
MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent),
        ui(new Ui::MainWindow)
{
    setAcceptDrops(true);


    progressDialog.reset(new QProgressDialog);
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

    dicom_parser dp(files,this);
    dp.exec();

}

void MainWindow::open_fib_at(int row,int col)
{
    loadFib(ui->recentFib->item(row,1)->text() + "/" +
            ui->recentFib->item(row,0)->text());
}

void MainWindow::open_src_at(int row,int col)
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
    check_prog(0,1);
    std::auto_ptr<ODFModel> new_handle(new ODFModel);
    if (!new_handle->load_from_file(&*file_name.begin()))
    {
        QMessageBox::information(this,"error",new_handle->fib_data.error_msg.c_str(),0);
        return;
    }
    tracking_window* new_mdi = new tracking_window(this,new_handle.release());
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->absolute_path = QFileInfo(filename).absolutePath();
    new_mdi->setWindowTitle(filename);
    new_mdi->showNormal();
    addFib(filename);
    add_work_dir(QFileInfo(filename).absolutePath());
}

void MainWindow::loadSrc(QStringList filenames)
{
    try
    {
        reconstruction_window* new_mdi = new reconstruction_window(filenames,this);
        new_mdi->setAttribute(Qt::WA_DeleteOnClose);
        new_mdi->show();
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
                                "Image files (*.dcm *.hdr *.nii *.nii.gz *.fdf 2dseq subject);;All files (*)" );
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
        for(unsigned int i = 1;1;++i)
        {
            image::io::bruker_info method_file;
            QString method_name = dir + "/" +QString::number(i)+"/method";
            if(!method_file.load_from_file(method_name.toLocal8Bit().begin()))
                break;
            if(!method_file["PVM_DwEffBval"].length())
                continue;
            filenames.push_back(dir + "/" +QString::number(i)+"/pdata/1/2dseq");
        }
        if(filenames.size() == 0)
        {
            QMessageBox::information(this,"Error","No diffusion data in this subject",0);
            return;
        }
        std::string file_name(subject_file["SUBJECT_study_name"]);
        file_name.erase(std::remove(file_name.begin(),file_name.end(),' '),file_name.end());
        dicom_parser dp(filenames,this);
        dp.set_name(dir + "/" + file_name.c_str() + ".src.gz");
        dp.exec();
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
    dicom_parser dp(filenames,this);
    dp.exec();
}

void MainWindow::on_Reconstruction_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(
                           this,
                           "Open Src files",
                           ui->workDir->currentText(),
                           "Src files (*.src.gz *.src);;All files (*)" );
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
                           "Fib files (*.fib.gz *.fib *.nii.gz *.nii);;All files (*)");
    if (filename.isEmpty())
        return;
    if(QFileInfo(filename).completeSuffix() == "nii" ||
            QFileInfo(filename).completeSuffix() == "nii.gz")
    {
        gz_nifti header;
        image::basic_image<float,3> I;
        std::vector<float> trans(16);
        trans[15] = 1.0;
        if(!header.load_from_file(filename.toLocal8Bit().begin()))
        {
            QMessageBox::information(this,"DSI Studio","Invalid file format",0);
            return;
        }
        header.toLPS(I);
        header.get_image_transformation(trans.begin());
        // from 0-based to 1-based
        trans[3] += 2.0;
        trans[7] += 2.0;
        trans[11] += 2.0;
        filename += ".mat";
        {
            image::io::mat_write mat(filename.toLocal8Bit().begin());
            mat << I;
            mat.write("voxel_size",header.nif_header.pixdim+1,3,1);
            mat.write("mni",&*trans.begin(),4,4);
        }
        loadFib(filename);
        return;
    }
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

void RenameDICOMToDir(QString FileName, QString ToDir)
{
    std::string person, sequence, imagename;
    {
        image::io::dicom header;
        if (!header.load_from_file(FileName.toLocal8Bit().begin()))
            return;

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
        QDir(ToDir).mkdir(ToDir);


    ToDir += "/";
    ToDir += Sequence;
    if (!QDir(ToDir).exists())
        QDir(ToDir).mkdir(ToDir);

    ToDir += "/";
    ToDir += ImageName;
    QFile(FileName).rename(FileName,ToDir);
}
void RenameDICOMUnderDirToDir(QString Dir, QString ToDir, bool include_sub)
{
    if(include_sub)
    {
        QStringList dirs = QDir(Dir).entryList(QStringList("*"),
                                                QDir::Dirs | QDir::NoSymLinks | QDir::NoDotAndDotDot);
        for(int index = 0;index < dirs.size();++index)
            RenameDICOMUnderDirToDir(Dir + "/" + dirs[index],ToDir,true);
    }
    QStringList files = QDir(Dir).entryList(QStringList("*"),
                            QDir::Files | QDir::NoSymLinks);
    for(int index = 0;index < files.size();++index)
        RenameDICOMToDir(Dir + "/" + files[index],ToDir);
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

void MainWindow::on_RenameDICOMDir_clicked()
{
    QString path =
        QFileDialog::getExistingDirectory(this,"Browse Directory",
                                          ui->workDir->currentText());
    if ( path.isEmpty() )
        return;
    RenameDICOMUnderDirToDir(path,path,true);
}

void MainWindow::on_vbc_clicked()
{
    VBCDialog* new_mdi = new VBCDialog(this,true);
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->show();
}

void MainWindow::on_averagefib_clicked()
{
    VBCDialog* new_mdi = new VBCDialog(this,false);
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
bool load_all_files(QStringList file_list,boost::ptr_vector<DwiHeader>& dwi_files);
QString get_src_name(QString file_name);
void MainWindow::on_batch_src_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                ui->workDir->currentText());
    if(dir.isEmpty())
        return;

    {
        QStringList dir_list;
        dir_list << dir;
        begin_prog("batch creating src");
        for(unsigned int i = 0;check_prog(i,dir_list.size());++i)
        {
            QDir cur_dir = dir_list[i];
            QStringList new_list = cur_dir.entryList(QStringList(""),QDir::AllDirs|QDir::NoDotAndDotDot);
            for(unsigned int index = 0;index < new_list.size();++index)
                dir_list << cur_dir.absolutePath() + "/" + new_list[index];

            QStringList dicom_file_list = cur_dir.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
            if(dicom_file_list.empty())
                continue;
            for (unsigned int index = 0;index < dicom_file_list.size();++index)
                dicom_file_list[index] = dir_list[i] + "/" + dicom_file_list[index];
            boost::ptr_vector<DwiHeader> dwi_files;
            if(!load_all_files(dicom_file_list,dwi_files))
                continue;
            if(prog_aborted())
                break;
            QString output = dir + "/" + QFileInfo(get_src_name(dicom_file_list[0])).baseName()+".src.gz";
            DwiHeader::output_src(output.toLocal8Bit().begin(),dwi_files,0);
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
    QString filename = QFileDialog::getOpenFileName(
                                this,
                                "Open Image",
                                ui->workDir->currentText(),
                                "image files (*.nii *.nii.gz *.dcm 2dseq)" );
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

void MainWindow::on_connectometry_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,
                           "Open Database files",
                           ui->workDir->currentText(),
                           "Database files (*.db.fib.gz);;All files (*)");
    if (filename.isEmpty())
        return;

    std::auto_ptr<vbc_database> database(new vbc_database);
    database.reset(new vbc_database);
    if(!database->load_database(filename.toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"Error","Invalid database format",0);
        return;
    }
    vbc_dialog* vbc = new vbc_dialog(this,database.release(),QFileInfo(filename).absoluteDir().absolutePath());
    vbc->setAttribute(Qt::WA_DeleteOnClose);
    vbc->show();
}

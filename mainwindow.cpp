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
#include <qmessagebox.h>
#include "reconstruction/reconstruction_window.h"
#include "dsi_interface_static_link.h"
#include "prog_interface_static_link.h"
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
#include "libs/tracking/fib_data.hpp"
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
    std::auto_ptr<FibData> new_handle(new FibData);
    if (!new_handle->load_from_file(&*file_name.begin()))
    {
        if(!prog_aborted())
            QMessageBox::information(this,"error",new_handle->error_msg.c_str(),0);
        return;
    }
    tracking_window* new_mdi = new tracking_window(this,new_handle.release());
    new_mdi->setAttribute(Qt::WA_DeleteOnClose);
    new_mdi->setWindowTitle(filename);
    new_mdi->showNormal();
    addFib(filename);
    add_work_dir(QFileInfo(filename).absolutePath());
    QDir::setCurrent(QFileInfo(filename).absolutePath());
}

void MainWindow::loadSrc(QStringList filenames)
{
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
        std::copy(header.nif_header.pixdim+1,header.nif_header.pixdim+4,vs);
    }
    if(QFileInfo(filename).baseName() == "2dseq")
    {
        image::io::bruker_2dseq bruker_header;
        if(!bruker_header.load_from_file(filename.toLocal8Bit().begin()))
        {
            QMessageBox::information(this,"DSI Studio","Invalid 2dseq format",0);
            return;
        }
        image::basic_image<float,3> I;
        float vs[3];
        bruker_header >> I;
        bruker_header.get_voxel_size(vs);
    }
    if(!I.empty())
    {
        std::auto_ptr<FibData> new_handle(new FibData);
        new_handle->mat_reader.add("dimension",I.geometry().begin(),3,1);
        new_handle->mat_reader.add("voxel_size",vs,3,1);
        new_handle->mat_reader.add("image",&*I.begin(),I.size(),1);
        new_handle->load_from_mat();
        new_handle->fib.index_name[0] = "image";
        new_handle->view_item[0].name = "image";
        tracking_window* new_mdi = new tracking_window(this,new_handle.release());
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
        set_title(QFileInfo(dirs[index]).baseName().toLocal8Bit().begin());
        for(unsigned int j = 0;j < files.size() && check_prog(index,dirs.size());++j)
        {
            set_title(files[j].toLocal8Bit().begin());
            RenameDICOMToDir(dirs[index] + "/" + files[j],path);
        }
    }
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
bool load_4d_nii(const char* file_name,boost::ptr_vector<DwiHeader>& dwi_files);
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


            boost::ptr_vector<DwiHeader> dwi_files;


            if(QFileInfo(dir_list[i] + "/data.nii.gz").exists() &&
               QFileInfo(dir_list[i] + "/bvals").exists() &&
               QFileInfo(dir_list[i] + "/bvecs").exists() &&
               load_4d_nii(QString(dir_list[i] + "/data.nii.gz").toLocal8Bit().begin(),dwi_files))
            {
                DwiHeader::output_src(QString(dir_list[i] + "/data.src.gz").toLocal8Bit().begin(),dwi_files,0);
                continue;
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
                std::copy(report.begin(),report.begin() + std::min<int>(80,report.length()+1),nii_out.nif_header.descrip);

                QString output_name = QFileInfo(get_src_name(dicom_file_list[0])).absolutePath() + "/" +
                                      QFileInfo(get_src_name(dicom_file_list[0])).baseName()+ "."+seq.c_str() + ".nii.gz";
                nii_out.save_to_file(output_name.toLocal8Bit().begin());
            }
            else
            {
                for(unsigned int index = 0;index < dwi_files.size();++index)
                    if(dwi_files[index].get_bvalue() < 100)
                    {
                        dwi_files[index].set_bvalue(0);
                        dwi_files[index].set_bvec(0,0,0);
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
    QString filename = QFileDialog::getOpenFileName(
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

void MainWindow::on_connectometry_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                           this,
                           "Open Database files",
                           ui->workDir->currentText(),
                           "Database files (*db?fib.gz);;All files (*)");
    if (filename.isEmpty())
        return;
    QDir::setCurrent(QFileInfo(filename).absolutePath());
    add_work_dir(QFileInfo(filename).absolutePath());
    std::auto_ptr<vbc_database> database(new vbc_database);
    database.reset(new vbc_database);
    begin_prog("reading connectometry db");
    if(!database->load_database(filename.toLocal8Bit().begin()))
    {
        QMessageBox::information(this,"Error","Invalid database format",0);
        return;
    }
    vbc_dialog* vbc = new vbc_dialog(this,database.release(),filename,true);
    vbc->setAttribute(Qt::WA_DeleteOnClose);
    vbc->show();
}

void MainWindow::on_workDir_currentTextChanged(const QString &arg1)
{
    QDir::setCurrent(arg1);
}

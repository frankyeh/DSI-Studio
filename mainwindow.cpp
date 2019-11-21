#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "tipl/tipl.hpp"
#include <QFileDialog>
#include <QDateTime>
#include <QUrl>
#include <QMessageBox>
#include <QProgressDialog>
#include <QDragEnterEvent>
#include <QMimeData>
#include <QAction>
#include <regtoolbox.h>
#include <qmessagebox.h>
#include "filebrowser.h"
#include "reconstruction/reconstruction_window.h"
#include "prog_interface_static_link.h"
#include "tracking/tracking_window.h"
#include "mainwindow.h"
#include "dicom/dicom_parser.h"
#include "ui_mainwindow.h"
#include "view_image.h"
#include "mapping/atlas.hpp"
#include "libs/gzip_interface.hpp"
#include "connectometry/group_connectometry_analysis.h"
#include "libs/tracking/fib_data.hpp"
#include "manual_alignment.h"
#include "connectometry/individual_connectometry.hpp"
#include "connectometry/createdbdialog.h"
#include "connectometry/db_window.h"
#include "connectometry/group_connectometry.hpp"
#include "connectometry/nn_connectometry.h"
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
std::vector<std::shared_ptr<tracking_window> > tracking_windows;
MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent),
        ui(new Ui::MainWindow)
{
    setAcceptDrops(true);
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


void MainWindow::closeEvent(QCloseEvent *event)
{
    auto windows_to_check = tracking_windows; // some windows may be removed
    for(size_t index = 0;index < windows_to_check.size();++index)
    {
        windows_to_check[index]->closeEvent(event);
        if(!event->isAccepted())
            return;
    }
    QMainWindow::closeEvent(event);
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
    tracking_windows.push_back(std::make_shared<tracking_window>(this,new_handle));
    tracking_windows.back()->setAttribute(Qt::WA_DeleteOnClose);
    tracking_windows.back()->setWindowTitle(filename);
    tracking_windows.back()->showNormal();
    tracking_windows.back()->resize(1200,700);
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
                                "Image files (*.dcm *.hdr *.nii *nii.gz *.fdf *.nhdr 2dseq subject);;All files (*)" );
    if ( filenames.isEmpty() )
        return;

    add_work_dir(QFileInfo(filenames[0]).absolutePath());
    if(QFileInfo(filenames[0]).completeBaseName() == "subject")
    {
        tipl::io::bruker_info subject_file;
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
                tipl::io::bruker_info method_file;
                QString method_name = dir + "/" +QString::number(i)+"/method";
                if(method_file.load_from_file(method_name.toLocal8Bit().begin()) &&
                   method_file["PVM_DwEffBval"].length())
                    is_dwi = true;
            }
            // has dif info in the imnd file
            {
                tipl::io::bruker_info imnd_file;
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
                           "Fib files (*fib.gz *.fib);;Image files (*nii.gz *.nii 2dseq);;All files (*)");
    if (filename.isEmpty())
        return;
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
        tipl::io::dicom header;
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


QStringList GetSubDir(QString Dir,bool recursive = true)
{
    QStringList sub_dirs;
    QStringList dirs = QDir(Dir).entryList(QStringList("*"),
                                            QDir::Dirs | QDir::NoSymLinks | QDir::NoDotAndDotDot);
    if(recursive)
        sub_dirs << Dir;
    for(int index = 0;index < dirs.size();++index)
    {
        QString new_dir = Dir + "/" + dirs[index];
        if(recursive)
            sub_dirs << GetSubDir(new_dir,recursive);
        else
            sub_dirs << new_dir;
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
    begin_prog("Renaming DICOM");
    for(unsigned int index = 0;check_prog(index,dirs.size());++index)
    {
        QStringList files = QDir(dirs[index]).entryList(QStringList("*"),
                                    QDir::Files | QDir::NoSymLinks);
        for(unsigned int j = 0;j < files.size() && index < dirs.size();++j)
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

bool load_all_files(QStringList file_list,std::vector<std::shared_ptr<DwiHeader> >& dwi_files);
bool find_bval_bvec(const char* file_name,QString& bval,QString& bvec);
bool load_4d_nii(const char* file_name,std::vector<std::shared_ptr<DwiHeader> >& dwi_files);
QString get_dicom_output_name(QString file_name,QString file_extension,bool add_path);

void MainWindow::on_batch_src_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                ui->workDir->currentText());
    if(dir.isEmpty())
        return;
    add_work_dir(dir);


    QString src_dir = dir + "/src";
    QString t1wt2w_dir = dir + "/t1wt2w";
    if((!QDir(src_dir).exists() && !QDir(src_dir).mkdir(src_dir)) ||
       (!QDir(t1wt2w_dir).exists() && !QDir(t1wt2w_dir).mkdir(t1wt2w_dir)))
    {
        QMessageBox::information(this,"Error","Cannot create folder",0);
        return;
    }

    QStringList sub_dir = QDir(dir).entryList(QStringList("*"),
                                            QDir::Dirs | QDir::NoSymLinks | QDir::NoDotAndDotDot);

    begin_prog("batch creating src");
    std::ofstream out((dir+"/log.txt").toStdString().c_str());
    for(int j = 0;check_prog(j,sub_dir.size()) && !prog_aborted();++j)
    {
        if(sub_dir[j] == "src" || sub_dir[j] == "t1wt2w")
            continue;
        out << "Processing " << sub_dir[j].toStdString() << std::endl;
        QString src_output_prefix = src_dir + "/" + sub_dir[j];
        QString t1wt2w_output_prefix = t1wt2w_dir + "/" + sub_dir[j];

        QStringList dir_list = GetSubDir(sub_dir[j],true);
        std::vector<std::shared_ptr<DwiHeader> > dicom_dwi_files;
        QString dicom_output;
        for(unsigned int i = 0;i < dir_list.size();++i)
        {
            QDir cur_dir = dir_list[i];
            // 4D nifti with same base name bvals and bvecs
            QStringList nifti_file_list = cur_dir.entryList(QStringList("*.nii.gz") << "*.nii",QDir::Files|QDir::NoSymLinks);
            for (unsigned int index = 0;index < nifti_file_list.size();++index)
            {
                out << "\tNIFTI file found at " << nifti_file_list[index].toStdString() << std::endl;
                std::vector<std::shared_ptr<DwiHeader> > dwi_files;
                if(!load_4d_nii(QString(dir_list[i] + "/" + nifti_file_list[index]).toLocal8Bit().begin(),dwi_files))
                {
                    out << "\t\tNot a 4D nifti. Skipping." << std::endl;
                    continue;
                }
                QString bval,bvec;
                if(!find_bval_bvec(QString(dir_list[i] + "/" + nifti_file_list[index]).toLocal8Bit().begin(),bval,bvec))
                {
                    out << "\t\tCannot find bval/bvec. Skipping." << std::endl;
                    continue;
                }
                if(!DwiHeader::has_b_table(dwi_files))
                {
                    out << "\t\tInvalid b-table. Skipping." << std::endl;
                    continue;
                }
                out << "\t\tSaving " << (src_output_prefix + ".src.gz").toStdString() << std::endl;
                DwiHeader::output_src(QString(src_output_prefix + ".src.gz").toLocal8Bit().begin(),dwi_files,0,false);
            }

            QStringList dicom_file_list = cur_dir.entryList(QStringList("*.dcm"),QDir::Files|QDir::NoSymLinks);
            if(dicom_file_list.empty())
                continue;
            out << "\tDICOM files found at " << dir_list[i].toStdString() << std::endl;
            for (unsigned int index = 0;index < dicom_file_list.size();++index)
                dicom_file_list[index] = dir_list[i] + "/" + dicom_file_list[index];

            std::vector<std::shared_ptr<DwiHeader> > dicom_files;
            if(!load_all_files(dicom_file_list,dicom_files) || prog_aborted())
                continue;
            if(dicom_files.size() == 1) //MPRAGE or T2W
            {
                out << "\t\tNot 4D DWI files." << std::endl;
                std::sort(dicom_file_list.begin(),dicom_file_list.end(),compare_qstring());
                tipl::io::volume v;
                tipl::io::dicom header;
                std::vector<std::string> file_list;
                for(unsigned int index = 0;index < dicom_file_list.size();++index)
                    file_list.push_back(dicom_file_list[index].toLocal8Bit().begin());
                if(!v.load_from_files(file_list,file_list.size()) ||
                   !header.load_from_file(dicom_file_list[0].toLocal8Bit().begin()))
                    continue;
                tipl::image<float,3> I;
                tipl::vector<3> vs;
                v >> I;
                v.get_voxel_size(vs);
                //non isotropic
                if((vs[0] != vs[1] || vs[0] != vs[2] || vs[1] != vs[2]) &&
                   (*std::min_element(vs.begin(),vs.end()))/(*std::max_element(vs.begin(),vs.end())) > 0.5f)
                {
                    float reso = *std::min_element(vs.begin(),vs.end());
                    tipl::vector<3,float> new_vs(reso,reso,reso);
                    tipl::image<float,3> J(tipl::geometry<3>(
                            int(std::ceil(float(I.width())*vs[0]/new_vs[0])),
                            int(std::ceil(float(I.height())*vs[1]/new_vs[1])),
                            int(std::ceil(float(I.depth())*vs[2]/new_vs[2]))));

                    tipl::transformation_matrix<float> T1;
                    T1.sr[0] = new_vs[0]/vs[0];
                    T1.sr[4] = new_vs[1]/vs[1];
                    T1.sr[8] = new_vs[2]/vs[2];
                    tipl::resample_mt(I,J,T1,tipl::cubic);
                    vs = new_vs;
                    I.swap(J);
                }
                gz_nifti nii_out;
                tipl::flip_xy(I);
                nii_out << I;
                nii_out.set_voxel_size(vs);


                std::string manu,make,seq,report,sequence;
                header.get_sequence_id(sequence);
                header.get_text(0x0008,0x0070,manu);//Manufacturer
                header.get_text(0x0008,0x1090,make);
                std::replace(manu.begin(),manu.end(),' ',(char)0);
                make.erase(std::remove(make.begin(),make.end(),' '),make.end());
                std::ostringstream info;
                info << manu.c_str() << " " << make.c_str() << " " << sequence
                    << ".TE=" << header.get_float(0x0018,0x0081) << ".TR=" << header.get_float(0x0018,0x0080)  << ".";
                report = info.str();
                if(report.size() < 80)
                    report.resize(80);
                nii_out.set_descrip(report.c_str());

                QString output = t1wt2w_output_prefix + "."+get_dicom_output_name(dicom_file_list[0],"",false)+"_"+sequence.c_str() + ".nii.gz";
                out << "\t\tConvert to NIFTI at " << output.toStdString() << std::endl;
                nii_out.save_to_file(output.toStdString().c_str());
            }
            else
            {
                out << "\t\t4D DWI files." << std::endl;
                if(dicom_dwi_files.empty() || dicom_dwi_files.front()->image.geometry() == dicom_files.front()->image.geometry())
                {
                    if(dicom_dwi_files.empty())
                        dicom_output = src_output_prefix + "." + get_dicom_output_name(dicom_file_list[0],".src.gz",false);
                    dicom_dwi_files.insert(dicom_dwi_files.end(),dicom_files.begin(),dicom_files.end());
                }
                else
                {
                    out << "[Warning] Two different DWI datasets are found: (" <<
                           dicom_dwi_files.front()->image.width() << "," <<
                           dicom_dwi_files.front()->image.height() << "," <<
                           dicom_dwi_files.front()->image.depth() << ") and (" <<
                           dicom_files.front()->image.width() << "," <<
                           dicom_files.front()->image.height() << "," <<
                           dicom_files.front()->image.depth() << "). Only the first one is used to generate SRC file." << std::endl;
                }
            }

        }

        if(dicom_dwi_files.size() > 1)
        {
            for(unsigned int index = 0;index < dicom_dwi_files.size();++index)
                if(dicom_dwi_files[index]->bvalue < 100.0f)
                {
                    dicom_dwi_files[index]->bvalue = 0.0f;
                    dicom_dwi_files[index]->bvec = tipl::vector<3>(0.0f,0.0f,0.0f);
                }
            out << "\tOutput DWI files to " << dicom_output.toStdString() << std::endl;
            DwiHeader::output_src(dicom_output.toLocal8Bit().begin(),dicom_dwi_files,0,false);
        }
        else
            out << "[Warning] No DWI data in this subject's folder. " << dicom_output.toStdString() << std::endl;
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

    loadSrc(search_files(dir,"*src.gz"));
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


bool MainWindow::load_db(std::shared_ptr<group_connectometry_analysis>& database,QString& filename)
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
    database = std::make_shared<group_connectometry_analysis>();
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
    std::shared_ptr<group_connectometry_analysis> database;
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
    std::shared_ptr<group_connectometry_analysis> database;
    if(!load_db(database,filename))
        return;
    group_connectometry* group_cnt = new group_connectometry(this,database,filename,true);
    group_cnt->setAttribute(Qt::WA_DeleteOnClose);
    group_cnt->show();
}


void MainWindow::on_run_cmd_clicked()
{
    if(!po.parse(ui->cmd_line->text().toStdString()))
    {
        QMessageBox::information(this,"Error",po.error_msg.c_str(),0);
        return;
    }
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

extern std::vector<std::string> fa_template_list,iso_template_list;
void MainWindow::on_ReconstructSRC_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(
                                this,
                                "Open directory",
                                ui->workDir->currentText());
    if(dir.isEmpty())
        return;

    QStringList list = search_files(dir,"*src.gz");

    for(int i = 0;i < list.size();++i)
    {
        std::shared_ptr<ImageModel> handle(std::make_shared<ImageModel>());
        if (!handle->load_from_file(list[i].toLocal8Bit().begin()))
        {
            QMessageBox::information(this,"error",QString("Cannot open ") +
                list[i] + " : " +handle->error_msg.c_str(),0);
            return;
        }

        handle->voxel.method_id = 7; // QSDR
        handle->voxel.param[0] = 1.25f;
        handle->voxel.ti.init(8); // odf order of 8
        handle->voxel.odf_xyz[0] = 0;
        handle->voxel.odf_xyz[1] = 0;
        handle->voxel.odf_xyz[2] = 0;
        handle->voxel.csf_calibration = 0;
        handle->voxel.max_fiber_number = 5;
        handle->voxel.r2_weighted = 0;
        handle->voxel.output_odf = true; // output ODF
        handle->voxel.check_btable = true;
        handle->voxel.output_tensor = false;
        handle->voxel.output_rdi = true;
        handle->voxel.thread_count = std::thread::hardware_concurrency();
        handle->voxel.primary_template = fa_template_list[0];
        handle->voxel.secondary_template = iso_template_list[0];

        //checking half shell
        {
            handle->voxel.half_sphere = handle->is_dsi_half_sphere();
            handle->voxel.scheme_balance = handle->need_scheme_balance();
        }

        const char* msg = handle->reconstruction();
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

bool load_image_from_files(QStringList filenames,tipl::image<float,3>& ref,tipl::vector<3>& vs,tipl::matrix<4,4,float>& trans);

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


    tipl::image<float,3> ref1,ref2;
    tipl::vector<3> vs1,vs2;
    tipl::matrix<4,4,float> t1,t2;
    if(!load_image_from_files(filename1,ref1,vs1,t1) ||
       !load_image_from_files(filename2,ref2,vs2,t2))
        return;
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,ref1,vs1,ref2,vs2,tipl::reg::affine,tipl::reg::mutual_info));
    manual->nifti_srow = t2;
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

void MainWindow::on_connectometry_nn_clicked()
{
    QString filename;
    std::shared_ptr<group_connectometry_analysis> database;
    if(!load_db(database,filename))
        return;
    nn_connectometry* nn_cnt = new nn_connectometry(this,database->handle,filename,true);
    nn_cnt->setAttribute(Qt::WA_DeleteOnClose);
    nn_cnt->show();
}

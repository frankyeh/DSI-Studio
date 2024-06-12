#include <QDir>
#include <QTimer>
#include <QMessageBox>
#include <QSettings>
#include <QFileDialog>
#include <QInputDialog>
#include "filebrowser.h"
#include "ui_filebrowser.h"
#include "dicom/dicom_parser.h"

FileBrowser::FileBrowser(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::FileBrowser)
{
    ui->setupUi(this);
    ui->graphicsView->setScene(&scene);
    ui->b_table->setColumnCount(7);
    ui->tableWidget->setColumnWidth(0,200);
    ui->tableWidget->setColumnWidth(1,90);
    ui->tableWidget->setColumnWidth(2,130);
    ui->tableWidget->setColumnWidth(3,60);
    ui->tableWidget->setColumnWidth(4,60);
    ui->tableWidget->setColumnWidth(5,60);
    ui->b_table->setColumnCount(4);
    ui->b_table->setColumnWidth(0,60);
    ui->b_table->setColumnWidth(1,80);
    ui->b_table->setColumnWidth(2,80);
    ui->b_table->setColumnWidth(3,80);

    ui->subject_list->setColumnWidth(0,150);
    ui->subject_list->setColumnWidth(1,200);
    ui->tableWidget->setStyleSheet("selection-background-color: blue");
    ui->subject_list->setStyleSheet("selection-background-color: blue");

    mon_map["Jan"] = "01";
    mon_map["Feb"] = "02";
    mon_map["Mar"] = "03";
    mon_map["Apr"] = "04";
    mon_map["May"] = "05";
    mon_map["Jun"] = "06";
    mon_map["Jul"] = "07";
    mon_map["Aug"] = "08";
    mon_map["Sep"] = "09";
    mon_map["Oct"] = "10";
    mon_map["Nov"] = "11";
    mon_map["Dec"] = "12";

    QSettings settings;
    ui->WorkDir->setText(settings.value("BW_WORK_PATH",QDir::currentPath()).toString());
    populateDirs();
    timer = new QTimer(this);
    timer->start(100);
    connect(timer, SIGNAL(timeout()), this, SLOT(show_image()));
}

FileBrowser::~FileBrowser()
{
    QSettings settings;
    settings.setValue("BW_WORK_PATH", ui->WorkDir->text());
    delete ui;
}



void FileBrowser::on_subject_list_currentCellChanged(int currentRow, int , int previousRow, int )
{
    if(currentRow == -1 || currentRow == previousRow)
        return;
    QDir directory = ui->WorkDir->text() + "/" +
                     ui->subject_list->item(currentRow,0)->text();
    // find all the acquisition
    data.clear();
    ui->tableWidget->clear();
    ui->tableWidget->setRowCount(0);
    image_list.clear();
    b_value_list.clear();
    b_vec_list.clear();
    if(QFileInfo(directory.absolutePath() + "/subject").exists())
    {
        QStringList params1,params2;
        params1 << "Method" << "PVM_EchoTime" << "PVM_RepetitionTime" << "PVM_SliceThick";
        params2 << "IMND_method_display_name" << "IMND_echo_time" << "IMND_rep_time" << "IMND_slice_thick";
        std::vector<unsigned char> seq_list;
        {
            QStringList seq_txt_list = directory.entryList(QStringList("*"),QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks);
            for(int index = 0;index < seq_txt_list.count();++index)
            {
                int seq = QString(seq_txt_list[index]).toInt();
                if(seq)
                    seq_list.push_back(seq);
            }
        }

        std::sort(seq_list.begin(),seq_list.end());
        for(int index = 0;index < seq_list.size();++index)
        {
            tipl::io::bruker_info method,acq,reco_file,d3proc_file;
            bool method_file = true;
            QString method_dir = directory.absolutePath() + "/" + QString::number(seq_list[index]);
            QString method_file_name =  method_dir+ "/method";
            QString acq_file_name =  method_dir+ "/acqp";
            if(!method.load_from_file(method_file_name.toStdString().c_str()))
            {
                method_file_name =  method_dir+ "/imnd";
                if(!method.load_from_file(method_file_name.toStdString().c_str()))
                    continue;
                method_file = false;
            }

            acq.load_from_file(acq_file_name.toStdString().c_str());
            // search for all reco
            QStringList params = method_file ? params1:params2;
            for(int reco = 1;;++reco)
            {
                QString reco_dir = method_dir + "/pdata/" + QString::number(reco);
                if(!QDir(reco_dir).exists())
                    break;
                QString cur_file_name = reco_dir + "/2dseq";
                QString cur_d3proc_name = reco_dir + "/d3proc";
                QString cur_reco_name = reco_dir + "/reco";
                if(!QFileInfo(cur_file_name).exists() ||
                   !reco_file.load_from_file(cur_reco_name.toStdString().c_str()))
                    continue;
                int row = ui->tableWidget->rowCount();
                ui->tableWidget->setRowCount(row+1);

                QStringList item_list;
                item_list << QString::number(seq_list[index])+":"+QString::number(reco) + ":" + method[params[0].toStdString().c_str()].c_str(); // sequence
                if(d3proc_file.load_from_file(cur_d3proc_name.toStdString().c_str()))
                    item_list << (d3proc_file["IM_SIX"] + " / " + d3proc_file["IM_SIY"] + " / "+ d3proc_file["IM_SIZ"]).c_str();
                else
                    item_list << reco_file["RECO_size"].c_str();

                // output resolution
                {
                    std::istringstream size_in(reco_file["RECO_size"].c_str());
                    std::istringstream fov_in(reco_file["RECO_fov"].c_str());
                    std::vector<float> data1((std::istream_iterator<float>(size_in)),std::istream_iterator<float>());
                    std::vector<float> data2((std::istream_iterator<float>(fov_in)),std::istream_iterator<float>());
                    std::ostringstream out;
                    for(int index = 0;index < data1.size() && index < data2.size();++index)
                        out << data2[index]*10.0/data1[index] << " / ";
                    out << method[params[3].toStdString().c_str()].c_str(); // slice thickness
                    item_list << out.str().c_str();
                }
                item_list << reco_file["RECO_fov"].c_str(); // FOV
                item_list << QString("%1/%2").
                             arg(method[params[1].toStdString().c_str()].c_str()).
                             arg(method[params[2].toStdString().c_str()].c_str()); // TE/TR
                item_list << acq["ACQ_flip_angle"].c_str(); //FA

                std::vector<float> b_value,b_vec;
                method.read("PVM_DwEffBval",b_value);
                method.read("PVM_DwGradVec",b_vec);
                b_vec.resize(b_value.size()*3);
                b_value_list.push_back(b_value);
                b_vec_list.push_back(b_vec);
                for(int col = 0;col < item_list.count();++col)
                    ui->tableWidget->setItem(row, col, new QTableWidgetItem(item_list[col]));
                image_list << cur_file_name;
            }
        }
        ui->tableWidget->setHorizontalHeaderLabels(QStringList() << "#:Sequence" << "Image size" << "Resolution(mm)" << "FOV(cm)" << "TE/TR" << "FA");
    }
    else
    {
        ui->tableWidget->setHorizontalHeaderLabels(QStringList() << "Name" << "Image size" << "Resolution(mm)" << " " << " " << " ");
        QStringList sub_dir_list = directory.entryList(QStringList("*"),QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks);
        for(unsigned int i = 0;i < sub_dir_list.size();++i)
        {
            QDir sub_dir = directory.absolutePath() + "/" + sub_dir_list[i];
            QStringList dcm_file_list = sub_dir.entryList(QStringList("*.dcm"),QDir::Files | QDir::NoSymLinks);

            if(!dcm_file_list.empty())
            {
                std::vector<std::string> dcm_file_list_full;
                for(unsigned int j = 0;j < dcm_file_list.size();++j)
                    dcm_file_list_full.push_back((sub_dir.absolutePath() + "/" + dcm_file_list[j]).toStdString().c_str());
                tipl::io::dicom_volume dcm;
                if(!dcm.load_from_files(dcm_file_list_full))
                    continue;
                int row = ui->tableWidget->rowCount();
                ui->tableWidget->setRowCount(row+1);
                QStringList item_list;
                item_list << sub_dir_list[i] + "(DICOM)";
                item_list << QString("%1/%2/%3").arg(dcm.shape().width()).arg(dcm.shape().height()).arg(dcm.shape().depth());
                tipl::vector<3,float> vs;
                dcm.get_voxel_size(vs);
                item_list << QString("%1/%2/%3").arg(vs[0]).arg(vs[1]).arg(vs[2]);
                for(int col = 0;col < item_list.count();++col)
                    ui->tableWidget->setItem(row, col, new QTableWidgetItem(item_list[col]));
                image_list << dcm_file_list_full[0].c_str();
                b_value_list.push_back(std::vector<float>());
                b_vec_list.push_back(std::vector<float>());
            }
        }
        QStringList nii_file_list = directory.entryList(QStringList() << "*.nii" << "*.nii.gz",QDir::Files | QDir::NoSymLinks);
        for(unsigned int i = 0;i < nii_file_list.size();++i)
        {
            QString file_name = directory.absolutePath() + "/" + nii_file_list[i];
            tipl::io::gz_nifti nii;
            if(!nii.load_from_file(file_name.toStdString()))
                continue;
            int row = ui->tableWidget->rowCount();
            ui->tableWidget->setRowCount(row+1);
            QStringList item_list;
            item_list << nii_file_list[i];
            item_list << QString("%1/%2/%3").arg(nii.width()).arg(nii.height()).arg(nii.depth());
            tipl::vector<3,float> vs;
            nii.get_voxel_size(vs);
            item_list << QString("%1/%2/%3").arg(vs[0]).arg(vs[1]).arg(vs[2]);
            for(int col = 0;col < item_list.count();++col)
                ui->tableWidget->setItem(row, col, new QTableWidgetItem(item_list[col]));
            image_list << file_name;
            b_value_list.push_back(std::vector<float>());
            b_vec_list.push_back(std::vector<float>());
        }
        QStringList mat_file_list = directory.entryList(QStringList() << "*.fib.gz" << "*.src.gz",QDir::Files | QDir::NoSymLinks);
        for(unsigned int i = 0;i < mat_file_list.size();++i)
        {
            QString file_name = directory.absolutePath() + "/" + mat_file_list[i];
            tipl::io::gz_mat_read mat;
            if(!mat.load_from_file(file_name.toStdString().c_str()))
                continue;
            int row = ui->tableWidget->rowCount();
            ui->tableWidget->setRowCount(row+1);
            QStringList item_list;
            item_list << mat_file_list[i];
            unsigned int r,c;
            tipl::shape<3> dim;
            tipl::vector<3> vs;
            if(!mat.read("dimension",dim) || !mat.read("voxel_size",vs))
                continue;

            item_list << QString("%1/%2/%3").arg(dim[0]).arg(dim[1]).arg(dim[2]);
            item_list << QString("%1/%2/%3").arg(vs[0]).arg(vs[1]).arg(vs[2]);
            for(int col = 0;col < item_list.count();++col)
                ui->tableWidget->setItem(row, col, new QTableWidgetItem(item_list[col]));
            image_list << file_name;
            b_value_list.push_back(std::vector<float>());
            b_vec_list.push_back(std::vector<float>());
            const float* b_table = 0;
            if(mat.read("b_table",r,c,b_table))
                for(unsigned int i = 0;i < r*c;i+=4)
                {
                    b_value_list.back().push_back(b_table[i]);
                    b_vec_list.back().push_back(b_table[i+1]);
                    b_vec_list.back().push_back(b_table[i+2]);
                    b_vec_list.back().push_back(b_table[i+3]);
                }

        }
    }

    if(ui->tableWidget->rowCount() > 0)
        ui->tableWidget->selectRow(0);
}

void FileBrowser::populateDirs(void)
{
    ui->subject_list->clear();
    ui->subject_list->setHorizontalHeaderLabels(QStringList()  << "Name" << "Date");
    ui->subject_list->setRowCount(0);

    ui->tableWidget->clear();
    QDir directory = ui->WorkDir->text();
    QStringList script_list = directory.entryList(QStringList("*"),QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks);
    ui->subject_list->setSortingEnabled(false);


    for(int index = 0;index < script_list.count();++index)
    {
        int row = ui->subject_list->rowCount();
        ui->subject_list->setRowCount(row+1);
        ui->subject_list->setItem(row, 0, new QTableWidgetItem(script_list[index]));


        QString subject_file_name = ui->WorkDir->text()+"/" + script_list[index] + "/subject";
        tipl::io::bruker_info subject_file;
        if(subject_file.load_from_file(subject_file_name.toStdString().c_str()))
        {
            std::istringstream in(subject_file["SUBJECT_date"]);
            std::string year,mon,date,time;
            in >> time >> date >> mon >> year;
            std::ostringstream out;
            out << year << "/" << mon_map[mon] << (date.size() == 1 ? "/0":"/") << date << " " << time;
            ui->subject_list->setItem(row, 1, new QTableWidgetItem(QString(subject_file["SUBJECT_name_string"].c_str()) + out.str().c_str()));
        }
        else
            ui->subject_list->setItem(row, 1, new QTableWidgetItem(" "));
    }
    ui->subject_list->setSortingEnabled(true);
    if(ui->subject_list->rowCount())
        ui->subject_list->selectRow(0);
}

void FileBrowser::on_changeWorkDir_clicked()
{
    QString filename = QFileDialog::getExistingDirectory(this,"Browse Directory",
                                              ui->WorkDir->text());
    if( filename.isEmpty() )
        return;
    ui->WorkDir->setText(filename);
    populateDirs();
}

void FileBrowser::show_image(void)
{
    /*
    if(preview_loaded && preview_thread.get())
    {
        preview_data.swap(data);
        cur_z = data.depth() >> 1;
        preview_loaded = false;
        preview_thread.reset(0);
    }

    if(!preview_file_name.isEmpty() && preview_thread.get() == 0)
    {
        preview_thread.reset(new std::future<void>(std::async(std::launch::async,&FileBrowser::preview_image,this,preview_file_name));
        preview_file_name.clear();
        preview_loaded = false;
    }*/
    if(!preview_file_name.isEmpty())
    {
        preview_image(preview_file_name);
        preview_data.swap(data);
        cur_z = data.depth() >> 1;
        preview_file_name.clear();
    }
    if(!data.size())
        return;
    ++cur_z;
    if(cur_z >= data.depth())
        cur_z = 0;

    {
        tipl::image<2,float> data_buffer;
        tipl::volume2slice(data,data_buffer,2,cur_z);
        slice_image.resize(data_buffer.shape());
        tipl::normalize_upper_lower2(data_buffer,slice_image,255);
    }
    view_image << slice_image;
    scene << view_image;

}
void FileBrowser::preview_image(QString file_name)
{
    preview_data.clear();
    if(QFileInfo(file_name).fileName() == "2dseq")
    {
        tipl::io::bruker_2dseq header;
        if(header.load_from_file(file_name.toStdString().c_str()))
        {
            header.get_image().swap(preview_data);
        }
        preview_loaded = true;
        return;
    }
    if(QFileInfo(file_name).completeSuffix() == "dcm")
    {
        QDir dir = QFileInfo(file_name).absoluteDir();
        QStringList dcm_file_list = dir.entryList(QStringList("*.dcm"),QDir::Files | QDir::NoSymLinks);
        std::vector<std::string> dcm_file_list_full;
        for(unsigned int j = 0;j < dcm_file_list.size();++j)
            dcm_file_list_full.push_back((dir.absolutePath() + "/" + dcm_file_list[j]).toStdString().c_str());
        tipl::io::dicom_volume dcm;
        if(!dcm.load_from_files(dcm_file_list_full))
            return;
        dcm >> preview_data;
        preview_loaded = true;
        return;
    }
    if(QFileInfo(file_name).suffix() == "gz")
    {
        tipl::io::gz_mat_read mat;
        if(mat.load_from_file(file_name.toStdString().c_str()))
        {
            tipl::shape<3> dim;
            if(mat.read("dimension",dim))
            {
                preview_data.resize(dim);
                if(!mat.read("fa0",preview_data))
                    mat.read("image0",preview_data);
            }
        }
        preview_loaded = true;
        return;
    }
    tipl::io::gz_nifti nii;
    if(nii.load_from_file(file_name.toStdString()))
        nii.get_untouched_image(preview_data);
    preview_loaded = true;
}

void FileBrowser::on_tableWidget_currentCellChanged(int currentRow, int, int previousRow, int)
{
    if(currentRow == previousRow || currentRow == -1)
        return;
    preview_file_name = image_list[currentRow];
    ui->b_table->clear();
    if(!b_value_list[currentRow].empty())
    {
        ui->b_table->setRowCount(b_value_list[currentRow].size());
        for(unsigned int i = 0;i < b_value_list[currentRow].size();++i)
        {
            ui->b_table->setItem(i,0, new QTableWidgetItem(QString::number(b_value_list[currentRow][i])));
            ui->b_table->setItem(i,1, new QTableWidgetItem(QString::number(b_vec_list[currentRow][i*3])));
            ui->b_table->setItem(i,2, new QTableWidgetItem(QString::number(b_vec_list[currentRow][i*3+1])));
            ui->b_table->setItem(i,3, new QTableWidgetItem(QString::number(b_vec_list[currentRow][i*3+2])));
        }
        ui->b_table->selectRow(0);
        ui->b_table->setHorizontalHeaderLabels(QStringList() << "b value" << "bx" << "by" << "bz");
    }
    setWindowTitle(preview_file_name);
    image_file_name = preview_file_name;
}

void FileBrowser::on_refresh_list_clicked()
{
    populateDirs();
}

void FileBrowser::on_create_src_clicked()
{
    QList<QTableWidgetItem *> sel = ui->tableWidget->selectedItems();
    if(!sel.size())
        return;
    QStringList filenames;
    std::set<int> selected_rows;
    for(unsigned int index = 0;index < sel.size();++index)
    {
        int row = sel[index]->row();
        if(selected_rows.find(row) != selected_rows.end())
            continue;
        selected_rows.insert(row);
        filenames << image_list[row];
    }
    if(filenames.empty())
        return;
    dicom_parser* dp = new dicom_parser(filenames,(QWidget*)parent());
    dp->set_name(QFileInfo(filenames[0]).absolutePath() + "/" + QFileInfo(filenames[0]).baseName() + ".src.gz");
    dp->setAttribute(Qt::WA_DeleteOnClose);
    dp->showNormal();
}

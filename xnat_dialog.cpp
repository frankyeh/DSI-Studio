#include <filesystem>
#include "xnat_dialog.h"
#include "ui_xnat_dialog.h"
#include <QMessageBox>
#include <QSettings>
#include <QFileDialog>
xnat_facade xnat_connection;

std::string jsonarray2tsv(QJsonArray data)
{
    if(!data.size())
        return std::string();
    std::string result;
    auto header = data[0].toObject().keys();
    for(auto& h : header)
    {
        if(!result.empty())
            result += "\t";
        result += h.toStdString();
    }
    std::multimap<QString,QString,std::greater<QString> > info;
    for(int i = 0;i < data.size();++i)
    {
        QString line;
        for(int j = 0;j < header.size();++j)
        {
            if(!line.isEmpty())
                line += "\t";
            line += data[i].toObject().value(header[j]).toString();
        }
        info.insert({data[i].toObject().value("date").toString(),line});
    }
    for(auto& item:info)
    {
        result += "\n";
        result += item.second.toStdString();
    }
    return result;
}

bool xnat_facade::good(void)
{
    if (cur_response && cur_response->error() != QNetworkReply::NoError)
    {
        error_msg = "connection error code ";
        error_msg += std::to_string(cur_response->error());
        if(cur_response->error() == 204)
            error_msg = "Authentication Failed";
        if(cur_response->error() == 11)
            error_msg = "Cannot login using guest";
        if(cur_response->error() == 3)
            error_msg = "XNAT Server not found";
        if(cur_response->error() == 6)
            error_msg = "SSL Failed. Please update SSL library";
        tipl::out() << error_msg;
        cur_response = nullptr;
    }
    return cur_response != nullptr;
}

void xnat_facade::clear(void)
{
    if(cur_response)
        {
            cur_response->abort();
            cur_response = nullptr;
        }
    prog = 0;
    total = 0;
    error_msg.clear();
}

void xnat_facade::get_html(std::string url,std::string auth)
{
    if(cur_response && cur_response->isRunning())
        cur_response->abort();
    download_prog = std::make_shared<tipl::progress>("downloading ",url.c_str());
    QNetworkRequest xnat_request(QUrl(url.c_str()));
    xnat_request.setRawHeader("Authorization", QString("Basic " + QString(auth.c_str()).toLocal8Bit().toBase64()).toLocal8Bit());
    error_msg.clear();
    cur_response = xnat_manager.get(xnat_request);
}
void check_name(std::string& name);
void xnat_facade::get_data(std::string site,std::string auth,
                                        const std::vector<std::string>& urls,
                                        std::string output_dir)
{
    if(site.back() == '/')
        site.pop_back();
    if(output_dir.back() == '/')
        output_dir.pop_back();

    total = urls.size();
    tipl::progress prog_("download data from ",site.c_str());
    tipl::out() << "a total of " << urls.size() << " files" << std::endl;
    for(prog = 0;prog_(prog,total);++prog)
    {
        std::string download_name = (output_dir+"/"+
                                     urls[prog].substr(urls[prog].find_last_of('/')+1)).c_str();
        if(std::filesystem::exists(download_name))
        {
            tipl::out() << "file exists, skipping: " << download_name.c_str() << std::endl;
            continue;
        }
        get_html(site + urls[prog],auth);
        {
            bool downloading = true;
            QObject::connect(cur_response, &QNetworkReply::finished,[this,&downloading]
            {
                download_prog.reset();
                downloading = false;
            });
            while(downloading)
                QApplication::processEvents();
        }
        if (!good())
            break;
        if(prog_.aborted())
        {
            error_msg = "download aborted";
            break;
        }

        tipl::out() << "saving " << download_name;
        QByteArray buf = cur_response->readAll();
        std::ofstream out(download_name.c_str(),std::ios::binary);
        if(!tipl::io::save_stream_with_prog(prog_,out,buf.begin(),buf.size(),error_msg))
        {
            std::remove(download_name.c_str());
            break;
        }
    }
    if(prog_.aborted())
        error_msg = "download aborted";
    cur_response = nullptr;
}

void xnat_facade::get_scans_data(std::string site,std::string auth,std::string experiment,std::string output_dir,std::string filter)
{
    if(site.back() == '/')
        site.pop_back();
    get_html(site + "/REST/experiments/" + experiment + "/scans/" + filter + "/files",auth);
    QObject::connect(cur_response, &QNetworkReply::finished,[=]{        
        std::vector<std::string> urls;
        if (good())
        {
            //auto const html = QString::fromUtf8(response->readAll());
            auto data = QJsonDocument::fromJson(cur_response->readAll()).object()["ResultSet"].toObject()["Result"].toArray();
            for(int i = 0;i < data.size();++i)
                urls.push_back(data[i].toObject().value("URI").toString().toStdString());
            tipl::out() << "a total of " << urls.size() << " files identified " << std::endl;
        }
        download_prog.reset();
        if(!urls.empty())
            get_data(site,auth,urls,output_dir);
    });
}
void xnat_facade::get_info(std::string site,std::string auth,std::string path)
{
    if(site.back() == '/')
        site.pop_back();
    get_html(site + path,auth);
    QObject::connect(cur_response, &QNetworkReply::finished,
    [this]{
       if (good())
       {
           tipl::out() << "receive content type: " << cur_response->header(QNetworkRequest::ContentTypeHeader).toString().toStdString() << std::endl;
           result = jsonarray2tsv(QJsonDocument::fromJson(cur_response->readAll()).object()["ResultSet"].toObject()["Result"].toArray());
       }
       cur_response = nullptr;
       download_prog.reset();
    });
}

xnat_dialog::xnat_dialog(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::xnat_dialog)
{
    ui->setupUi(this);
    ui->download->hide();
    ui->download_settings->hide();

    QSettings settings;
    ui->url->setText(settings.value("xnat_url",ui->url->text()).toString());
    ui->username->setText(settings.value("xnat_username").toString());
    ui->password->setText(settings.value("xnat_password").toString());
    ui->save_dir->setText(settings.value("xnat_save_dir",QDir::current().path()).toString());

    ui->sub_dir->addItem("/");
    for(int i = 0;i < experiment_header.size();++i)
        ui->sub_dir->addItem(QString("/{%1}").arg(experiment_header[i]));
    ui->sub_dir->setCurrentIndex(settings.value("xnat_sub_dir",3).toInt());
}

xnat_dialog::~xnat_dialog()
{
    delete ui;
}
void xnat_dialog::experiment_view_updated()
{
    if(xnat_connection.is_running())
    {
        if(ui->statusbar->currentMessage().isEmpty())
            ui->statusbar->showMessage("Loading");
        ui->statusbar->showMessage(ui->statusbar->currentMessage()+".");
        return;
    }
    ui->statusbar->showMessage("");
    ui->connect->setText("Connect");
    experiment_view_timer.reset();

    if(xnat_connection.has_error())
    {
        QMessageBox::critical(this,"ERROR",xnat_connection.error_msg.c_str());
        return;
    }
    if(xnat_connection.result.empty())
    {
        QMessageBox::critical(this,"ERROR","No available experiment data to download");
        return;
    }
    {
        QSettings settings;
        settings.setValue("xnat_url",ui->url->text());
        settings.setValue("xnat_username",ui->username->text());
        settings.setValue("xnat_password",ui->password->text());
    }

    // update project list
    {
        std::istringstream in(xnat_connection.result);
        std::string line;
        std::getline(in,line); // header
        int project_index = QString(line.c_str()).split('\t').indexOf("project");
        if(project_index != -1)
        {
            QStringList project_list;
            while(std::getline(in,line))
            {
                QString project(QString(line.c_str()).split('\t')[project_index]);
                if(project_list.indexOf(project) == -1)
                    project_list << project;
            }
            ui->project_list->clear();
            ui->project_list->addItem("All");
            ui->project_list->addItems(project_list);
            ui->project_list->setCurrentRow(0);
        }
    }

    ui->download->show();
    ui->download_settings->show();
}
void xnat_dialog::on_project_list_currentRowChanged(int)
{
    if(!ui->project_list->count() ||
        xnat_connection.is_running() ||
        !ui->project_list->currentItem())
        return;

    int header_width[] = {120,120,120,200,150};
    std::istringstream in(xnat_connection.result);
    std::string line;
    std::getline(in,line);
    QStringList header = QString(line.c_str()).split('\t');
    int project_index = header.indexOf("project");
    std::vector<int> header_seq(size_t(experiment_header.size()));
    for(size_t index = 0;index < header_seq.size();++index)
        header_seq[index] = std::max<int>(0,header.indexOf(experiment_header[int(index)]));

    if(!ui->experiment_list->columnCount())
    {
        ui->experiment_list->setColumnCount(experiment_header.size());
        ui->experiment_list->setHorizontalHeaderLabels(experiment_header);
        for(int index = 0;index < experiment_header.size();++index)
            ui->experiment_list->setColumnWidth(index,header_width[index]);
    }
    int row_count = 0;
    ui->experiment_list->setRowCount(0);
    while(std::getline(in,line))
    {
        auto data = QString(line.c_str()).split('\t');
        if(ui->project_list->currentRow() != 0 && project_index != -1 &&
           data[project_index] != ui->project_list->currentItem()->text())
            continue;
        ++row_count;
        ui->experiment_list->setRowCount(row_count);
        for(size_t i = 0;i < header_seq.size();++i)
            ui->experiment_list->setItem(row_count-1,int(i),new QTableWidgetItem(data[header_seq[i]]));
    }

}

void xnat_dialog::on_connect_clicked()
{
    if(!ui->url->text().contains("."))
    {
        QMessageBox::critical(this,"ERROR","Please assign server URL");
        return;
    }
    if(ui->connect->text() == "Abort")
    {
        ui->connect->setText("Connect");
        experiment_view_timer.reset();
        ui->statusbar->showMessage("");
        return;
    }
    ui->project_list->clear();
    ui->experiment_list->setRowCount(0);
    ui->connect->setText("Abort");
    QString auth = ui->username->text() + ":" + ui->password->text();
    xnat_connection.get_experiments_info(ui->url->text().toStdString(),auth == ":" ? std::string():auth.toStdString());
    experiment_view_timer = std::make_shared<QTimer>();
    connect(experiment_view_timer.get(), SIGNAL(timeout()), this, SLOT(experiment_view_updated()));
    experiment_view_timer->setInterval(1000);
    experiment_view_timer->start();

}

void xnat_dialog::on_open_dir_clicked()
{
    QString filename = QFileDialog::getExistingDirectory(this,"Browse Directory",ui->save_dir->text());
    if ( filename.isEmpty() )
        return;
    ui->save_dir->setText(filename);
}

void xnat_dialog::on_experiment_list_itemSelectionChanged()
{
    if(xnat_connection.is_running())
        return;
    if(ui->experiment_list->selectionModel()->selectedRows().size())
        ui->download->setEnabled(true);
    else
        ui->download->setEnabled(false);
}

void xnat_dialog::on_download_clicked()
{
    xnat_connection.clear();

    if(ui->download->text() == "Abort")
    {
        download_timer.reset();        
        ui->experiment_list->setEnabled(true);
        ui->connection_group->setEnabled(true);
        ui->download_settings->setEnabled(true);
        ui->project_experiement_group->setEnabled(true);

        cur_download_index = 0;

        ui->statusbar->showMessage("");
        ui->download->setText("Download");
        return;
    }
    ui->experiment_list->setEnabled(false);
    ui->connection_group->setEnabled(false);
    ui->download_settings->setEnabled(false);
    ui->project_experiement_group->setEnabled(false);
    ui->download->setText("Abort");

    cur_download_index = -1;
    output_dirs.clear();

    download_timer = std::make_shared<QTimer>();
    connect(download_timer.get(), SIGNAL(timeout()), this, SLOT(download_status()));
    download_timer->setInterval(1000);
    download_timer->start();

    {
        QSettings settings;
        settings.setValue("xnat_save_dir",ui->save_dir->text());
        settings.setValue("xnat_sub_dir",ui->sub_dir->currentIndex());
    }

}
QStringList rename_dicom_at_dir(QString path,QString output);
void dicom2src_and_nii(std::string dir_);
void xnat_dialog::download_status()
{
    if(xnat_connection.has_error())
    {
        QMessageBox::critical(this,"ERROR",xnat_connection.error_msg.c_str());
        on_download_clicked(); // abort();
        return;
    }
    if(!xnat_connection.is_running())
    {
        ++cur_download_index;
        if(cur_download_index >= ui->experiment_list->selectionModel()->selectedRows().size())
        {
            on_download_clicked();
            QMessageBox::information(this,QApplication::applicationName(),"Download Completed");
            if(QMessageBox::information(this,QApplication::applicationName(),"Rename DICOM files?",QMessageBox::Yes|QMessageBox::No) == QMessageBox::Yes)
            {
                QStringList subject_dirs;
                {
                    tipl::progress prog("Renaming DICOM");
                    for(size_t i = 0;prog(i,output_dirs.size());++i)
                        subject_dirs << rename_dicom_at_dir(output_dirs[i].c_str(),output_dirs[i].c_str());
                    if(prog.aborted())
                        return;
                }
                if(QMessageBox::information(this,QApplication::applicationName(),"Convert DICOM to SRC/NII?",QMessageBox::Yes|QMessageBox::No) == QMessageBox::Yes)
                {
                    tipl::progress prog("Converting DICOM to SRC/NII?");
                    for(size_t i = 0;prog(i,subject_dirs.size());++i)
                        dicom2src_and_nii(subject_dirs[i].toStdString());
                    if(prog.aborted())
                        return;
                }
            }
            return;
        }
        QString output_dir = ui->save_dir->text();
        if(ui->sub_dir->currentIndex() > 0)
        {
            if(output_dir.back() != '/' && output_dir.back() != '/')
                output_dir += "/";

            std::string sub_dir = ui->experiment_list->item(ui->experiment_list->selectionModel()->selectedRows()[cur_download_index].row(),
                                                            ui->sub_dir->currentIndex()-1)->text().toStdString();
            check_name(sub_dir);
            output_dir += sub_dir.c_str();
            if(!QDir(output_dir).exists())
            {
                if(!QDir(output_dir).mkdir(output_dir))
                {
                    QMessageBox::critical(this,"ERROR",QString("Cannot create directory %1").arg(output_dir));
                    on_download_clicked(); // abort();
                    return;
                }
            }
        }
        QString auth = ui->username->text() + ":" + ui->password->text();
        output_dirs.push_back(output_dir.toStdString());
        xnat_connection.get_scans_data(ui->url->text().toStdString(),
                                             auth == ":" ? std::string():auth.toStdString(),
                                             ui->experiment_list->item(ui->experiment_list->selectionModel()->selectedRows()[cur_download_index].row(),4/*ID*/)->text().toStdString(),
                                             output_dir.toStdString());

    }
    ui->statusbar->showMessage(QString("Downloading %1 (%2/%3) ").arg(ui->experiment_list->item(ui->experiment_list->selectionModel()->selectedRows()[cur_download_index].row(),2/*label*/)->text()).arg(xnat_connection.prog).arg(xnat_connection.total));
}


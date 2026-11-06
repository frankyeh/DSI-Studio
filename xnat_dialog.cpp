#include <filesystem>
#include "xnat_dialog.h"
#include "ui_xnat_dialog.h"
#include <QMessageBox>
#include <QSettings>
#include <QFileDialog>


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
        error_msg = showQNetworkReplyError(cur_response).toStdString();
        cur_response = nullptr;
    }
    return cur_response != nullptr;
}

void xnat_facade::clear(void)
{
    if(cur_response)
        std::exchange(cur_response,nullptr)->abort(); // cleared first, so its finished handler sees a cancel
    error_msg.clear();
}

xnat_facade::xnat_facade(void)
{
    downloads.get = [this](const QString& url){return QSharedPointer<QNetworkReply>(get(url.toStdString()),&QObject::deleteLater);};
    downloads.on_finished = [this](const QString& error){error_msg = error.toStdString();};
}

QNetworkReply* xnat_facade::get(const std::string& url)
{
    QNetworkRequest xnat_request(QUrl(url.c_str()));
    xnat_request.setAttribute(QNetworkRequest::Http2AllowedAttribute,false);
    xnat_request.setRawHeader("Authorization", QString("Basic " + QString(auth.c_str()).toLocal8Bit().toBase64()).toLocal8Bit());
    return xnat_manager.get(xnat_request);
}

void xnat_facade::get_html(std::string url,std::string auth)
{
    if(cur_response && cur_response->isRunning())
        cur_response->abort();
    download_prog = std::make_shared<tipl::progress>("downloading ",url.c_str());
    this->auth = auth;
    error_msg.clear();
    cur_response = get(url);
}
void check_name(std::string& name);
void xnat_facade::get_scans_data(std::string site,std::string auth,std::vector<std::pair<std::string,std::string> > experiments,std::string filter)
{
    if(experiments.empty())
        return;
    if(site.back() == '/')
        site.pop_back();
    get_html(site + "/REST/experiments/" + experiments[0].first + "/scans/" + filter + "/files",auth);
    QObject::connect(cur_response, &QNetworkReply::finished,[=]{
        download_prog.reset();
        if(!good())
        {
            downloads.error = QString::fromStdString(error_msg);
            return downloads.next();
        }
        auto data = QJsonDocument::fromJson(cur_response->readAll()).object()["ResultSet"].toObject()["Result"].toArray();
        tipl::out() << "a total of " << data.size() << " files identified " << std::endl;
        for(int i = 0;i < data.size();++i)
        {
            auto uri = data[i].toObject().value("URI").toString();
            downloads.add("xnat",site.c_str()+uri,QDir(experiments[0].second.c_str()).filePath(uri.mid(uri.lastIndexOf('/')+1)),
                          false,experiments[0].first.c_str());
        }
        cur_response = nullptr;
        get_scans_data(site,auth,{experiments.begin()+1,experiments.end()},filter);
        downloads.next();
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

    xnat_connection.downloads.on_progress = [this](const QString& id){xnat_connection.downloads.paint(ui->experiment_list,4/*ID*/,2/*label*/,id);};
    xnat_connection.downloads.on_change = [this](void){on_experiment_list_itemSelectionChanged();};
    xnat_connection.downloads.on_finished = [this](const QString& error)
    {
        if(xnat_connection.cur_response) // more experiments still being listed: report at the end
            return void(xnat_connection.downloads.error = error);
        if(error.isEmpty())
            QMessageBox::information(this,QApplication::applicationName(),"Download Completed");
        else
            QMessageBox::critical(this,"ERROR",error);
    };
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
    xnat_connection.downloads.on_progress(QString());

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
    bool downloading = xnat_connection.is_running();
    ui->connection_group->setEnabled(!downloading);
    ui->download_settings->setEnabled(!downloading);
    ui->download->setEnabled(downloading || ui->experiment_list->selectionModel()->selectedRows().size());
    ui->download->setText(downloading ? "Cancel" : "Download");
}

void xnat_dialog::on_download_clicked()
{
    if(xnat_connection.is_running())
    {
        xnat_connection.clear();
        xnat_connection.downloads.cancel("xnat");
        return;
    }

    std::vector<std::pair<std::string,std::string> > experiments;
    for(const auto& index : ui->experiment_list->selectionModel()->selectedRows())
    {
        QString output_dir = ui->save_dir->text();
        if(ui->sub_dir->currentIndex() > 0)
        {
            std::string sub_dir = ui->experiment_list->item(index.row(),ui->sub_dir->currentIndex()-1)->text().toStdString();
            check_name(sub_dir);
            output_dir = QDir(output_dir).filePath(sub_dir.c_str());
            if(!QDir().mkpath(output_dir))
                return QMessageBox::critical(this,"ERROR",QString("Cannot create directory %1").arg(output_dir)),void();
        }
        experiments.push_back({ui->experiment_list->item(index.row(),4/*ID*/)->text().toStdString(),output_dir.toStdString()});
    }

    {
        QSettings settings;
        settings.setValue("xnat_save_dir",ui->save_dir->text());
        settings.setValue("xnat_sub_dir",ui->sub_dir->currentIndex());
    }

    QString auth = ui->username->text() + ":" + ui->password->text();
    xnat_connection.downloads.clear_finished();
    ui->experiment_list->clearSelection(); // the selection color would hide the progress bars
    xnat_connection.get_scans_data(ui->url->text().toStdString(),auth == ":" ? std::string():auth.toStdString(),experiments);
    on_experiment_list_itemSelectionChanged();
}

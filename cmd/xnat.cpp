#include <QtNetwork>
#include <iostream>
#include "program_option.hpp"
#include <QApplication>
#include "tipl/tipl.hpp"
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
bool RenameDICOMToDir(QString FileName, QString ToDir,QString& NewName);
class xnat_facade{
private:
    bool response_good(QNetworkReply* response)
    {
        if (response->error() != QNetworkReply::NoError)
        {
            error_msg = "connection error code ";
            error_msg += std::to_string(response->error());
            has_error = true;
            running = false;
            return false;
        }
        return true;
    }
    void clear(void)
    {
        progress = 0;
        total = 0;
        terminated = false;
        running = false;
        has_error = false;
        error_msg.clear();
        result.clear();
    }
public:
    QNetworkAccessManager xnat_manager;
    bool terminated = false;
    bool running = false;
    bool has_error = false;
    size_t progress = 0;
    size_t total = 0;
    std::string result,error_msg;
    QNetworkReply* get_html(std::string url,std::string auth)
    {
        std::cout << "request " << url << std::endl;
        QNetworkRequest xnat_request(QUrl(url.c_str()));
        xnat_request.setAttribute(QNetworkRequest::FollowRedirectsAttribute, true);
        xnat_request.setRawHeader("Authorization", QString("Basic " + QString(auth.c_str()).toLocal8Bit().toBase64()).toLocal8Bit());
        return xnat_manager.get(xnat_request);
    }
public:
    void get_experiments_dicom(std::string site,std::string auth,const std::vector<std::string>& dicom_urls,std::string output_dir,bool overwrite)
    {
        if(site.back() == '/')
            site.pop_back();
        total = dicom_urls.size();
        for(progress = 0;progress < dicom_urls.size() && !terminated;++progress)
        {
            std::cout << "downloading " << dicom_urls[progress] << std::endl;
            auto* response = get_html(site + dicom_urls[progress],auth);
            if (!response_good(response))
                return;
            bool dicom_downloading = true;
            QObject::connect(response, &QNetworkReply::finished,[&dicom_downloading]
            {
                dicom_downloading = false;
            });
            while(dicom_downloading)
            {
                QApplication::processEvents();
                if(terminated)
                {
                    response->abort();
                    running = false;
                    return;
                }
            }
            if(!response_good(response))
                break;
            QString dicom_file_name = (output_dir+"/download").c_str();
            std::shared_ptr<QFile> file(new QFile);
            file->setFileName(dicom_file_name);
            if (!file->open(QIODevice::WriteOnly))
            {
                error_msg = "no write permission at ";
                error_msg += output_dir;
                has_error = true;
                break;
            }
            file->write(response->readAll());
            file->close();
            QString NewName;
            if(!RenameDICOMToDir(dicom_file_name,output_dir.c_str(),NewName))
            {
                error_msg = "cannot create folders at ";
                error_msg += output_dir;
                has_error = true;
                break;
            }
            if(QFileInfo(NewName).exists() && overwrite)
                QFile::remove(NewName);
            if(QFileInfo(NewName).exists())
                std::cout << "file exist at " << NewName.toStdString() << std::endl;
            else
            {
                std::cout << "saving " << NewName.toStdString() << std::endl;
                if(!QFile::rename(dicom_file_name,NewName))
                {
                    error_msg = "cannot rename dicom file to ";
                    error_msg += NewName.toStdString();
                    has_error = true;
                    break;
                }
            }
        }
        running = false;
    }
    void get_experiments_data(std::string site,std::string auth,std::string experiment,std::string output_dir,bool overwrite)
    {
        clear();
        auto* response = get_html(site + "REST/experiments/" + experiment + "/scans/ALL/files",auth);
        if (!response_good(response))
            return;
        running = true;
        QObject::connect(response, &QNetworkReply::finished,[=]{
            if (!response_good(response))
                return;
            std::cout << "receive content type: " << response->header(QNetworkRequest::ContentTypeHeader).toString().toStdString() << std::endl;
            //auto const html = QString::fromUtf8(response->readAll());
            auto data = QJsonDocument::fromJson(response->readAll()).object()["ResultSet"].toObject()["Result"].toArray();
            std::vector<std::string> dicom_urls;
            for(int i = 0;i < data.size();++i)
                dicom_urls.push_back(data[i].toObject().value("URI").toString().toStdString());
            std::cout << "a total of " << dicom_urls.size() << " files" << std::endl;
            get_experiments_dicom(site,auth,dicom_urls,output_dir,overwrite);
            running = false;
        });
    }
    void get_experiments_info(std::string site,std::string auth)
    {
        clear();
        auto* response = get_html(site + "REST/experiments/",auth);
        if (!response_good(response))
            return;
        running = true;
        QObject::connect(response, &QNetworkReply::finished,
        [response,this]{
           if (!response_good(response))
               return;
           std::cout << "receive content type: " << response->header(QNetworkRequest::ContentTypeHeader).toString().toStdString() << std::endl;
           result = jsonarray2tsv(QJsonDocument::fromJson(response->readAll()).object()["ResultSet"].toObject()["Result"].toArray());
           running = false;
        });
    }

} xnat_connection;


int xnat(program_option& po)
{
    std::string site = po.get("site");
    std::string auth = po.get("auth");
    std::string experiment = po.get("experiment");
    std::string output = po.get("output");
    if(site.empty() || auth.empty())
    {
        std::cout << "ERROR: plase specify xnat server using --site=https://xnat.my.edu and login information using --auth=username:password" << std::endl;
        return 1;
    }

    if(QFileInfo(output.c_str()).isDir() && output.back() != '\\' && output.back() != '/')
        output += '/';

    if(experiment.empty())
    {
        if(output.empty() || QFileInfo(output.c_str()).isDir())
            output = "experiments.txt";
        if(!QString(output.c_str()).endsWith(".txt"))
            output += ".txt";
        std::cout << "writing output to " << output << std::endl;
        xnat_connection.get_experiments_info(site,auth);
    }
    else
    {
        if(output.empty())
            output = QDir::current().path().toStdString();
        if(!QFileInfo(output.c_str()).isDir())
        {
            std::cout << "ERROR: please specify output directory using --output" << std::endl;
            return 1;
        }
        std::cout << "writing output to " << output << std::endl;
        xnat_connection.get_experiments_data(site,auth,experiment,output,po.get("overwrite",0));
    }

    while(xnat_connection.running)
        QApplication::processEvents();

    if (xnat_connection.has_error)
    {
        std::cout << "ERROR: " << xnat_connection.error_msg << std::endl;
        return 1;
    }

    if(experiment.empty())
    {
        std::cout << "write experiment info to " << output << std::endl;
        std::ofstream(output) << xnat_connection.result;
    }
    else
    {
        std::cout << "experiment data saved to " << output << std::endl;
    }
    return 0;
}

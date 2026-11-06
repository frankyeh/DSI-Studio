#ifndef XNAT_DIALOG_H
#define XNAT_DIALOG_H
#include <iostream>
#include <QtNetwork>
#include <QApplication>
#include <QMainWindow>
#include <QTimer>
#include "zlib.h"
#include "TIPL/tipl.hpp"
#include "fiber_data_hub.hpp" // download_queue

class xnat_facade{
public:
    QNetworkAccessManager xnat_manager;
    QNetworkReply* cur_response = nullptr;
    std::shared_ptr<tipl::progress> download_prog;
    std::string auth,result,error_msg;
    download_queue downloads; // scan files, keyed by experiment id
public:
    xnat_facade(void);
    QNetworkReply* get(const std::string& url);
    void get_html(std::string url,std::string auth);
    // experiments: id and output directory, listed one after another while their files download
    void get_scans_data(std::string site,std::string auth,std::vector<std::pair<std::string,std::string> > experiments,std::string filter = "ALL");
public:
    void get_info(std::string site,std::string auth,std::string path);
    void get_experiments_info(std::string site,std::string auth)
    {get_info(site,auth,"/REST/experiments/");}
    void get_scans_info(std::string site,std::string auth,std::string experiment_id)
    {get_info(site,auth,std::string("/REST/experiments/") + experiment_id + "/scans");}
    void get_resources_info(std::string site,std::string auth,std::string experiment_id)
    {get_info(site,auth,std::string("/REST/experiments/") + experiment_id + "/resources");}
public:
    bool good(void);
    void clear(void);
    bool is_running(void)const{return cur_response || downloads.running;}
    bool has_error(void)const{return !error_msg.empty();}
};

namespace Ui {
class xnat_dialog;
}

class xnat_dialog : public QMainWindow
{
    Q_OBJECT

public:
    xnat_facade xnat_connection;
    explicit xnat_dialog(QWidget *parent = nullptr);
    ~xnat_dialog();
private:
    QStringList experiment_header = {"date","project","label","insert_date","ID"};
    std::shared_ptr<QTimer> experiment_view_timer;
private slots:
    void on_connect_clicked();
    void experiment_view_updated();

    void on_open_dir_clicked();

    void on_experiment_list_itemSelectionChanged();

    void on_download_clicked();

    void on_project_list_currentRowChanged(int currentRow);

private:
    Ui::xnat_dialog *ui;
};

#endif // XNAT_DIALOG_H

#ifndef XNAT_DIALOG_H
#define XNAT_DIALOG_H
#include <QtNetwork>
#include <iostream>
#include <QApplication>
#include "TIPL/tipl.hpp"
#include <QMainWindow>
#include <QTimer>

class xnat_facade{
public:
    QNetworkAccessManager xnat_manager;
    QNetworkReply* cur_response = nullptr;
    std::shared_ptr<tipl::progress> download_prog;
    size_t prog = 0;
    size_t total = 0;
    std::string result,error_msg;
public:
    void get_html(std::string url,std::string auth);
    void get_data(std::string site,std::string auth,const std::vector<std::string>& urls,std::string output_dir);
    void get_scans_data(std::string site,std::string auth,std::string experiment,std::string output_dir,std::string filter = "ALL");
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
    bool is_running(void)const{return cur_response;}
    bool has_error(void)const{return !error_msg.empty();}
};
extern xnat_facade xnat_connection;

namespace Ui {
class xnat_dialog;
}

class xnat_dialog : public QMainWindow
{
    Q_OBJECT

public:
    explicit xnat_dialog(QWidget *parent = nullptr);
    ~xnat_dialog();
private:
    QStringList experiment_header = {"date","project","label","insert_date","ID"};
    std::shared_ptr<QTimer> experiment_view_timer,download_timer;
    int cur_download_index = 0;
    std::vector<std::string> output_dirs;
private slots:
    void on_connect_clicked();
    void exeriment_view_updated();
    void download_status();

    void on_open_dir_clicked();

    void on_experiment_list_itemSelectionChanged();

    void on_download_clicked();

    void on_project_list_currentRowChanged(int currentRow);

private:
    Ui::xnat_dialog *ui;
};

#endif // XNAT_DIALOG_H

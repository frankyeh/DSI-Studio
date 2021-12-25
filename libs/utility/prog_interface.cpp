#include <QProgressDialog>
#include <QApplication>
#include <QObject>
#include <memory>
#include <ctime>
#include <iostream>
#include <chrono>
#include <thread>
#include <QTime>
#include "prog_interface_static_link.h"
bool has_gui = false;
std::shared_ptr<QProgressDialog> progressDialog;
static std::vector<QTime> process_time,t_last;
bool prog_aborted_ = false;
auto start_time = std::chrono::high_resolution_clock::now();
std::vector<std::string> progress::status_list;
std::vector<std::string> progress::at_list;
std::thread::id main_thread_id = std::this_thread::get_id();
bool is_main_thread(void)
{
    return main_thread_id == std::this_thread::get_id();
}

inline bool processing_time_less_than(int time)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::high_resolution_clock::now() - start_time).count() < time;
}
void progress::update_prog(bool show_now)
{
    if(!show_now && processing_time_less_than(250))
        return;
    if(!progressDialog.get())
        progressDialog.reset(new QProgressDialog(get_status().c_str(),"Cancel",0,100));
    else
        progressDialog->setLabelText(get_status().c_str());

    progressDialog->show();
    if(!progressDialog->property("raised").toInt())
    {
        progressDialog->setWindowFlags(Qt::WindowStaysOnTopHint);
        progressDialog->raise();
        progressDialog->activateWindow();
        progressDialog->setProperty("raised",1);
    }
    QApplication::processEvents();
}
std::string progress::get_status(void)
{
    std::string result;
    for(size_t i = 0;i < status_list.size();++i)
    {
        if(i)
            result += "\n";
        result += status_list[i];
        if(i < at_list.size())
        {
            result += " ";
            result += at_list[i];
        }
    }
    return result;
}
void progress::begin_prog(bool show_now)
{
    if(!has_gui || !is_main_thread())
        return;
    start_time = std::chrono::high_resolution_clock::now();
    process_time.resize(status_list.size());
    process_time.back().start();
    t_last.resize(status_list.size());
    t_last.back().start();
    prog_aborted_ = false;
    if(progressDialog.get())
        progressDialog->setProperty("raised",0);
    update_prog(show_now);
}

void progress::show(const char* status,bool show_now)
{
    std::cout << status << std::endl;
    if(status_list.empty())
        return;
    status_list.back() = status;
    if(!has_gui || !is_main_thread())
        return;
    update_prog(show_now);
}
progress::~progress(void)
{
    status_list.pop_back();
    process_time.pop_back();
    t_last.pop_back();
    if(!has_gui || !is_main_thread())
        return;
    if(!status_list.empty())
    {
        update_prog();
        return;
    }
    at_list.clear();
    if(progressDialog.get())
    {
        prog_aborted_ = progressDialog->wasCanceled();
        progressDialog->close();
        progressDialog.reset();
    }
    QApplication::processEvents();
}
bool progress::check_prog(unsigned int now,unsigned int total)
{
    if(!has_gui || !is_main_thread() || status_list.empty())
        return now < total;
    if(now >= total)
    {
        if(at_list.size() == status_list.size())
            at_list.back().clear();
        return false;
    }
    if(t_last.back().elapsed() > std::min<int>(int(total)*50,500))
    {
        t_last.back().start();
        int expected_sec = (process_time.back().elapsed()*int(total-now)/int(now+1)/1000/60);
        at_list.resize(status_list.size());
        at_list.back() = QString("(%1/%2)").arg(now).arg(total).toStdString();
        if(expected_sec)
            at_list.back() += QString(" %1 min").arg(expected_sec).toStdString();
        update_prog();
        if(progressDialog.get())
        {
            progressDialog->setRange(0, int(total));
            progressDialog->setValue(int(now));
            if(progressDialog->wasCanceled())
                return false;
            QApplication::processEvents();
        }
    }
    return now < total;
}

bool progress::aborted(void)
{
    if(!has_gui || !is_main_thread())
        return false;
    if(progressDialog.get())
        return progressDialog->wasCanceled();
    if(prog_aborted_)
        return true;
    return false;
}




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
QTime t_total,t_last;
bool prog_aborted_ = false;
auto start_time = std::chrono::high_resolution_clock::now();
std::vector<std::string> progress::status_list;
std::thread::id main_thread_id = std::this_thread::get_id();
bool is_main_thread(void)
{
    return main_thread_id == std::this_thread::get_id();
}
void progress::update_prog(bool show_now)
{
    if(!show_now && std::chrono::duration_cast<std::chrono::milliseconds>(
       std::chrono::high_resolution_clock::now() - start_time).count() < 250)
        return;
    if(!progressDialog.get())
        progressDialog.reset(new QProgressDialog(get_status().c_str(),"Cancel",0,100));
    else
        progressDialog->setLabelText(get_status().c_str());

    progressDialog->show();
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
    }
    return result;
}
void progress::begin_prog(bool show_now)
{
    if(!has_gui || !is_main_thread())
        return;
    start_time = std::chrono::high_resolution_clock::now();
    t_total.start();
    t_last.start();
    prog_aborted_ = false;
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
    if(!has_gui || !is_main_thread())
        return;
    if(!status_list.empty())
    {
        update_prog();
        return;
    }
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
    if(!progressDialog.get())
        update_prog();
    if(!progressDialog.get())
        return now < total;
    if(now >= total || progressDialog->wasCanceled())
        return false;
    if(now == 0 || now == total)
        t_total.start();
    if(t_last.elapsed() > 500)
    {
        t_last.start();
        long expected_sec = 0;
        if(now)
            expected_sec = ((double)t_total.elapsed()*(double)(total-now)/(double)now/1000.0);
        progressDialog->setRange(0, total);
        progressDialog->setValue(now);
        QString label = get_status().c_str();
        if(expected_sec)
            progressDialog->setLabelText(label + QString("\n%1 of %2\n%3 min %4 sec").
                                             arg(now).arg(total).arg(expected_sec/60).arg(expected_sec%60));
        else
            progressDialog->setLabelText(label + QString("\n%1 of %2...").arg(now).arg(total));
        progressDialog->show();
        QApplication::processEvents();
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




#include <QProgressDialog>
#include <QApplication>
#include <QObject>
#include <memory>
#include <ctime>
#include <iostream>
#include <QTime>

std::auto_ptr<QProgressDialog> progressDialog;
QTime t_total,t_last;
bool lock_dialog = false;

void begin_prog(const char* title,bool lock)
{
    if(!progressDialog.get())
    {
        std::cout << title << std::endl;
        return;
    }
    lock_dialog = lock;
    progressDialog.reset(new QProgressDialog(title,"Cancel",0,100,0));
    progressDialog->show();
    QApplication::processEvents();
    t_total.start();
    t_last.start();
}
bool is_running(void)
{
    if(!progressDialog.get())
        return false;
    return progressDialog->isVisible();
}

void set_title(const char* title)
{
    if(!progressDialog.get())
        return;
    progressDialog->setWindowTitle(title);
    QApplication::processEvents();
}
int check_prog(int now,unsigned int total)
{
    if(now == total && progressDialog.get() && !lock_dialog)
    {
        progressDialog.reset(new QProgressDialog("","Cancel",0,10,0));
        QApplication::processEvents();
        return false;
    }
    if(now == 0 || now == total)
        t_total.start();
    if(progressDialog.get() && (t_last.elapsed() > 500))
    {
        t_last.start();
        long expected_sec = 0;
        if(now)
            expected_sec = ((double)t_total.elapsed()*(double)(total-now)/(double)now/1000.0);
        if(progressDialog->wasCanceled())
            return false;
        progressDialog->setRange(0, total);
        progressDialog->setValue(now);
        if(expected_sec)
            progressDialog->setLabelText(QString("%1 of %2, estimated time: %3 min %4 sec").
                                             arg(now).arg(total).arg(expected_sec/60).arg(expected_sec%60));
        else
            progressDialog->setLabelText(QString("%1 of %2...").arg(now).arg(total));
        QApplication::processEvents();
    }
    return now < total;
}

int prog_aborted(void)
{
    if(progressDialog.get())
        return progressDialog->wasCanceled();
    return false;
}




#include <QProgressDialog>
#include <QtGui/QApplication>
#include <QObject>
#include <memory>
#include <ctime>
#include <iostream>
#include <QTime>
std::auto_ptr<QProgressDialog> progressDialog;
QTime t_total,t_last;
bool lock_dialog = false;

        extern "C" void begin_prog(const char* title,bool lock)
        {
            if(!progressDialog.get())
            {
                std::cout << title << std::endl;
                return;
            }
            lock_dialog = lock;
            progressDialog->show();
            progressDialog->setWindowTitle(title);
            qApp->processEvents();
            t_total.start();
        }
        extern "C" void end_prog(void)
        {
            lock_dialog = false;
            if(progressDialog.get())
                progressDialog.reset(new QProgressDialog("","Abort",0,10,0));
        }

        extern "C" void set_title(const char* title)
        {
            if(!progressDialog.get())
                return;
            progressDialog->setWindowTitle(title);
            qApp->processEvents();
        }

        extern "C" void can_cancel(int cancel)
        {
            if(cancel && progressDialog.get())
                progressDialog->setCancelButtonText("&Cancel");
        }
        extern "C" int check_prog(int now,int total)
        {
            if(now == total && progressDialog.get() && !lock_dialog)
            {
                progressDialog.reset(new QProgressDialog("","Abort",0,10,0));
                qApp->processEvents();
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
                qApp->processEvents();
            }
            return now < total;
        }

        extern "C" int prog_aborted(void)
        {
            if(progressDialog.get())
                return progressDialog->wasCanceled();
            return false;
        }




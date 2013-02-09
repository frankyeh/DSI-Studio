#include <QProgressDialog>
#include <QtGui/QApplication>
#include <QObject>
#include <memory>
#include <ctime>
#include <iostream>

std::auto_ptr<QProgressDialog> progressDialog;
long cur_time = 0;
long zero_time = 0;
const long interval = 500;

        extern "C" void begin_prog(const char* title)
        {
            if(!progressDialog.get())
                return;
            progressDialog.reset(new QProgressDialog("initializing...","Abort",0,10,0));
            progressDialog->setMinimumDuration(0);
            progressDialog->setWindowTitle(title);
            qApp->processEvents();
            cur_time = zero_time = 0;
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
            long now_time = std::clock();
            if(now == total && progressDialog.get())
                progressDialog.reset(new QProgressDialog("","Abort",0,10,0));
            if(now == 0 || now == total)
                zero_time = now_time;
            if(progressDialog.get() && (now_time - cur_time > interval))
            {

                long expected_sec = 0;
                if(now != 0 && now_time != zero_time)
                    expected_sec = ((double)(now_time-zero_time)*(double)(total-now)/(double)now/(double)CLOCKS_PER_SEC);
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
                cur_time = std::clock();
            }
            return now < total;
        }

        extern "C" int prog_aborted(void)
        {
            if(progressDialog.get())
                return progressDialog->wasCanceled();
            return false;
        }




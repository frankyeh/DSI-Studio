#ifndef CONSOLE_H
#define CONSOLE_H
#include <mutex>
#include <iostream>
#include <QMainWindow>
#include <QTimer>

class QTextEdit;
class console_stream :  public std::basic_streambuf<char>
{
    std::basic_streambuf<char>* cout_buf = nullptr;
public:
    console_stream(void):std::basic_streambuf<char>(){}
protected:
    virtual int_type overflow(int_type v) override;
    virtual std::streamsize xsputn(const char *p, std::streamsize n) override;
public:
    std::mutex edit_buf;
    QString buf;
    QTextEdit* log_window = nullptr;
    bool has_output = false;
    void attach(void)
    {
        if(!cout_buf)
            cout_buf = std::cout.rdbuf(this);
    }
    void detach(void)
    {
        if(cout_buf)
            std::cout.rdbuf(cout_buf);
    }
};
extern console_stream console;


namespace Ui {
class Console;
}

class Console : public QMainWindow
{
    Q_OBJECT
public:
    explicit Console(QWidget *parent = nullptr);
    ~Console();

private slots:

    void on_set_dir_clicked();
    void check_msg(void);
private:
    Ui::Console *ui;
    QTimer *timer;
    bool eventFilter(QObject *obj, QEvent *event) override;
protected:
    void closeEvent(QCloseEvent *event) override;
};

#endif // CONSOLE_H

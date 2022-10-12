#include <QFileDialog>
#include <QDir>
#include <QMessageBox>
#include "console.h"
#include "ui_console.h"
#include "program_option.hpp"
#include "TIPL/tipl.hpp"
#include "prog_interface_static_link.h"

console_stream console;

void console_stream::show_output(void)
{
    if(!tipl::is_main_thread<0>() || !log_window || !has_output)
        return;
    QStringList strSplitted;
    {
        std::lock_guard<std::mutex> lock(edit_buf);
        strSplitted = buf.split("\n");
        buf = strSplitted.back();
    }
    for(int i = 0; i+1 < strSplitted.size(); i++)
        log_window->append(strSplitted.at(i));
    QApplication::processEvents();
    has_output = false;
}
std::basic_streambuf<char>::int_type console_stream::overflow(std::basic_streambuf<char>::int_type v)
{
    {
        std::lock_guard<std::mutex> lock(edit_buf);
        buf.push_back(char(v));
    }

    if (v == '\n')
    {
        has_output = true;
        show_output();
    }
    return v;
}

std::streamsize console_stream::xsputn(const char *p, std::streamsize n)
{
    std::lock_guard<std::mutex> lock(edit_buf);
    buf += p;
    return n;
}

Console::Console(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Console)
{
    ui->setupUi(this);
    ui->pwd->setText(QString("[%1]$ ./dsi_studio ").arg(QDir().current().absolutePath()));
    console.log_window = ui->console;
    console.show_output();

}

Console::~Console()
{
    console.log_window = nullptr;
    delete ui;
}

int rec(program_option& po);
int trk(program_option& po);
int src(program_option& po);
int ana(program_option& po);
int exp(program_option& po);
int atl(program_option& po);
int cnt(program_option& po);
int vis(program_option& po);
int ren(program_option& po);
int atk(program_option& po);
int reg(program_option& po);
int xnat(program_option& po);

int run_action_with_wildcard(program_option& po);
void Console::on_run_cmd_clicked()
{
    program_option po;
    if(ui->cmd_line->text().startsWith("dsi_studio "))
        ui->cmd_line->setText(ui->cmd_line->text().remove("dsi_studio "));
    if(!po.parse(ui->cmd_line->text().toStdString()))
    {
        QMessageBox::critical(this,"ERROR",po.error_msg.c_str());
        return;
    }
    if (!po.has("action"))
    {
        std::cout << "ERROR: invalid command, use --help for more detail" << std::endl;
        return;
    }
    run_action_with_wildcard(po);
}


void Console::on_set_dir_clicked()
{
    QString dir =
        QFileDialog::getExistingDirectory(this,"Browse Directory","");
    if ( dir.isEmpty() )
        return;
    QDir::setCurrent(dir);
    ui->pwd->setText(QString("[%1]$ ./dsi_studio ").arg(QDir().current().absolutePath()));

}


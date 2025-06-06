#include <QFileDialog>
#include <QDir>
#include <QMessageBox>
#include <QProcess>
#include <QKeyEvent>
#include <QTextCursor>
#include "console.h"
#include "ui_console.h"
#include "zlib.h"
#include "TIPL/tipl.hpp"

console_stream console;

// Function to map ANSI color codes to Qt colors
QColor colorFromAnsiCode(int code) {
    switch (code) {
        case 31: return QColor(161,67,76);
        case 32: return QColor(67,161,76);
        case 33: return QColor(207,151,45);
        case 34: return QColor(95,168,253);
        case 35: return QColor(179,137,243);
        default: return QColor(230,237,243);
    }
}

// Function to append text to QTextEdit with ANSI color and bold support
void appendColoredText(QTextEdit &textEdit, const QString &text)
{
    QTextCursor cursor = textEdit.textCursor();
    cursor.movePosition(QTextCursor::End);

    QString currentPart;
    int endIndex = -1;
    for(int index = 0;index < text.length();++index)
    {
        // Found start of escape sequence, find end of sequence
        if (text[index] == '\033' && index + 1 < text.length() && text[index + 1] == '[' &&
            (endIndex = text.indexOf('m', index)) != -1)
        {
            if (!currentPart.isEmpty())
            {
                cursor.insertText(currentPart);
                currentPart.clear();
            }

            // Get the ANSI escape sequence
            QString escapeSequence = text.mid(index, endIndex - index + 1);
            QStringList parts = escapeSequence.mid(2, endIndex - index - 2).split(';');
            if (parts.isEmpty())
                // Invalid escape sequence, treat as regular text
                currentPart += text.mid(index, endIndex - index + 1);
            else
            {
                QTextCharFormat format;
                for (const QString& part : parts)
                {
                    int code = part.toInt();
                    if (code == 1)
                        format.setFontWeight(QFont::Bold);
                    else
                        format.setForeground(colorFromAnsiCode(code));
                }

                // Move cursor to end and apply format
                cursor.movePosition(QTextCursor::End);
                cursor.setCharFormat(format);
            }
            // Move index to end of escape sequence
            index = endIndex;
            continue;

        }
        currentPart.push_back(text[index]);// Regular text
    }

    // Append any remaining regular text
    currentPart.push_back("\n");
    cursor.insertText(currentPart);
}


std::basic_streambuf<char>::int_type console_stream::overflow(std::basic_streambuf<char>::int_type v)
{
    std::lock_guard<std::mutex> lock(edit_buf);
    buf.push_back(char(v));
    if (v == '\n')
        has_output = true;
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
    ui->console->ensureCursorVisible();
    ui->pwd->setText(QString("[%1]$ ./dsi_studio ").arg(QDir().current().absolutePath()));
    console.log_window = ui->console;
    ui->cmd_line->installEventFilter(this);
    ui->cmd_line->addItem("--action=rec --source=*.sz");
    ui->cmd_line->addItem("--action=trk --source=*.fz");
    ui->cmd_line->addItem("--action=atk --source=*.fz");


    check_msg();
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(check_msg()));
    timer->setInterval(1000);
    timer->start();

}
void Console::check_msg(void)
{
    if(!console.has_output)
        return;
    console.has_output = false;
    QStringList strSplitted;
    {
        std::lock_guard<std::mutex> lock(console.edit_buf);
        strSplitted = console.buf.split('\n');
        console.buf = strSplitted.back();
    }

    QScrollBar *vScrollBar = ui->console->verticalScrollBar();
    bool shouldAutoScroll = false;
    if (vScrollBar)// If the scrollbar is not visible, or if its value is at maximum, set auto-scroll flag.
        shouldAutoScroll = (!vScrollBar->isVisible() || vScrollBar->value() == vScrollBar->maximum());

    for(int i = 0; i+1 < strSplitted.size(); i++)
        appendColoredText(*(ui->console),strSplitted[i]);

    if (shouldAutoScroll && vScrollBar) // Scroll to bottom after appending text.
        vScrollBar->setValue(vScrollBar->maximum());
}

Console::~Console()
{
    console.log_window = nullptr;
    delete ui;
}

int run_action_with_wildcard(tipl::program_option<tipl::out>& po,int ac, char *av[]);

void Console::on_set_dir_clicked()
{
    QString dir =
        QFileDialog::getExistingDirectory(this,"Browse Directory","");
    if ( dir.isEmpty() )
        return;
    QDir::setCurrent(dir);
    ui->pwd->setText(QString("[%1]$ ./dsi_studio ").arg(QDir().current().absolutePath()));

}
void Console::closeEvent(QCloseEvent *event)
{
    event->ignore();
    hide();
}

bool Console::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == ui->cmd_line && event->type() == QEvent::KeyPress)
    {
        QKeyEvent *keyEvent = static_cast<QKeyEvent*>(event);
        if (keyEvent->key() == Qt::Key_Return || keyEvent->key() == Qt::Key_Enter)
        {
            ui->pwd->setText(QString("[%1]$ ./dsi_studio ").arg(QDir().current().absolutePath()));
            QString text = ui->cmd_line->currentText();
            tipl::program_option<tipl::out> po;
            if(ui->cmd_line->currentText().isEmpty())
                return true;
            if(ui->cmd_line->currentText().startsWith("dsi_studio "))
                ui->cmd_line->setCurrentText(ui->cmd_line->currentText().remove("dsi_studio "));
            if(!ui->cmd_line->currentText().startsWith("--"))
            {
                QProcess process;
                #ifdef Q_OS_WIN
                    process.start("cmd.exe", QStringList() << "/c" << ui->cmd_line->currentText());
                #else
                    process.start(ui->cmd_line->currentText());
                #endif
                process.waitForFinished(); // Wait for the process to finish
                ui->console->append(QString::fromUtf8(process.readAllStandardOutput()));
                ui->pwd->setText(QString("[%1]$ ./dsi_studio ").arg(QDir().current().absolutePath()));
                ui->cmd_line->addItem(ui->cmd_line->currentText());
                ui->cmd_line->setCurrentText(QString());
                return true;

            }
            if(!po.parse(ui->cmd_line->currentText().toStdString()))
            {
                QMessageBox::critical(this,"ERROR",po.error_msg.c_str());
                return true;
            }
            if (!po.has("action"))
                po.set("action","vis");
            run_action_with_wildcard(po,0,nullptr);
            ui->cmd_line->insertItem(0,ui->cmd_line->currentText());
            ui->cmd_line->setCurrentText(QString());
            return true; // Consume the event
        }
    }
    return QObject::eventFilter(obj, event);
}



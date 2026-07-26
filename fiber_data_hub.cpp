#include <QDir>
#include <QDate>
#include <QDateTime>
#include <QCoreApplication>
#include <QEventLoop>
#include <QFile>
#include <QFileDialog>
#include <QHeaderView>
#include <QInputDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QNetworkReply>
#include <QProgressDialog>
#include <QRegularExpression>
#include <QSignalBlocker>
#include <QStandardPaths>
#include <QTextStream>
#include <QTimer>
#include <QThread>
#include <QUrlQuery>
#include <QVBoxLayout>
#include <set>

#include "fiber_data_hub.hpp"
#include "ui_fiber_data_hub.h"
#include "mainwindow.h"
#include "connectometry/db_window.h"
#include "connectometry/group_connectometry.hpp"
#include "connectometry/group_connectometry_analysis.h"

FiberDataHub::FiberDataHub(MainWindow* parent):
    QMainWindow(parent),main_window(*parent),ui(new Ui::FiberDataHub)
{
    ui->setupUi(this);
    ui->github_release_note->setCurrentIndex(0);
    ui->github_open_file->setVisible(false);
    ui->github_open_file_mode->setVisible(false);
    ui->download_dir->setText(main_window.work_dir());
    initialize();
}

FiberDataHub::~FiberDataHub()
{
    delete ui;
}

void FiberDataHub::initialize()
{
    if(fetch_github)
        return;
    fetch_github = true;

    QString content = settings.value("hub_content").toString();

    QString url = main_window.fiber_data_hub_url();
    if(!url.isEmpty())
    {
        auto reply = main_window.get(url);

        if(content.isEmpty())
        {
            QEventLoop loop;
            connect(reply.get(),&QNetworkReply::finished,&loop,&QEventLoop::quit);
            loop.exec();

            if(reply->error() == QNetworkReply::NoError)
                settings.setValue("hub_content",content = QString::fromUtf8(reply->readAll()));
        }
        else
        {
            connect(reply.get(),&QNetworkReply::finished,this,[this,reply]()
                    {
                        if(reply->error() == QNetworkReply::NoError)
                            settings.setValue("hub_content",QString::fromUtf8(reply->readAll()));
                    });
        }
    }

    QString md,line;
    md.reserve(content.size());

    QSignalBlocker block(ui->github_repo);
    QTextStream in(&content);
    const QString mark = "](https://github.com/";

    for(bool first = true;in.readLineInto(&line);first = false)
    {
        if(first)
            continue;

        int p = 0;
        while(p < line.size() && line[p].isSpace())
            ++p;
        if(line.mid(p).startsWith("<img src"))
            continue;

        md += line + "\n";

        if(!line.startsWith("- "))
            continue;

        int m = line.indexOf(mark);
        if(m < 0)
            continue;

        int b = line.lastIndexOf('[',m);
        int r = m + mark.size();
        int s = line.indexOf('/',r);
        int e = s < 0 ? -1 : line.indexOf('/',s + 1);
        if(e < 0 && s >= 0)
            e = line.indexOf(')',s + 1);

        if(b >= 0 && e > s)
            ui->github_repo->addItem(line.mid(b + 1,m - b - 1),line.mid(r,e - r));
    }

    ui->github_note->setMarkdown(md);
    ui->github_note->setReadOnly(true);
    ui->github_note->setOpenExternalLinks(true);
    on_github_repo_currentIndexChanged(0);
}

bool FiberDataHub::command(const std::vector<std::string>& cmd)
{
    error_msg.clear();
    auto fail = [this](std::string error)
    {
        error_msg = error;
        return false;
    };
    const std::string usage =
        "hub repos | hub tags <repo> | hub files <repo> <tag> [text] [offset] [limit] | "
        "hub open <repo> <tag> <file> | hub download <repo> <tag> <file> <dir>";

    if(cmd.size() < 2 || cmd[1] == "help")
        return tipl::out() << usage,true;
    initialize();
    if(!ui->github_repo->count())
    {
        fetch_github = false;
        return fail("Fiber Data Hub is not ready; retry");
    }
    if(cmd[1] == "repos")
    {
        for(int row = 0;row < ui->github_repo->count();++row)
            tipl::out() << row << "\t" << ui->github_repo->itemData(row).toString().toStdString();
        return true;
    }
    if(cmd.size() < 3)
        return fail(usage);

    int repo = ui->github_repo->findData(QString::fromUtf8(cmd[2]));
    if(repo < 0)
        return fail("repository not found");
    ui->github_repo->setCurrentIndex(repo);
    on_github_repo_currentIndexChanged(repo);
    if(cmd[1] == "tags")
    {
        if(!ui->github_tags->rowCount())
            return fail("repository data is loading; retry");
        for(int row = 0;row < ui->github_tags->rowCount();++row)
            tipl::out() << row << "\t" << ui->github_tags->item(row,0)->text().toStdString();
        return true;
    }
    if(cmd.size() < 4)
        return fail(usage);

    int tag = -1;
    for(int row = 0;row < ui->github_tags->rowCount();++row)
        if(ui->github_tags->item(row,0)->text() == QString::fromUtf8(cmd[3]))
            tag = row;
    if(tag < 0)
        return fail("tag not found or still loading");
    ui->github_tags->setCurrentCell(tag,0);
    on_github_tags_itemSelectionChanged();
    if(cmd[1] == "files")
    {
        QString text = cmd.size() > 4 ? QString::fromUtf8(cmd[4]) : QString();
        bool ok = true;
        int offset = cmd.size() > 5 ? QString::fromUtf8(cmd[5]).toInt(&ok) : 0;
        if(!ok || offset < 0)
            return fail("invalid offset");
        int limit = cmd.size() > 6 ? QString::fromUtf8(cmd[6]).toInt(&ok) :
                        ui->github_release_files->rowCount();
        if(!ok || limit < 0)
            return fail("invalid limit");
        for(int row = 0;row < ui->github_release_files->rowCount() && limit;++row)
        {
            if(!ui->github_release_files->item(row,0)->text().contains(text,Qt::CaseInsensitive))
                continue;
            if(offset)
            {
                --offset;
                continue;
            }
            tipl::out() << row << "\t"
                        << ui->github_release_files->item(row,0)->text().toStdString() << "\t"
                        << ui->github_release_files->item(row,1)->text().toStdString() << "\t"
                        << QFile::exists(QStandardPaths::writableLocation(QStandardPaths::TempLocation) +
                                         "/" + cur_tag + "/" +
                                         ui->github_release_files->item(row,0)->text());
            --limit;
        }
        return true;
    }
    if(cmd.size() < 5)
        return fail(usage);

    int file = -1;
    for(int row = 0;row < ui->github_release_files->rowCount();++row)
        if(ui->github_release_files->item(row,0)->text() == QString::fromUtf8(cmd[4]))
            file = row;
    if(file < 0)
        return fail("file not found");
    ui->github_release_files->setCurrentCell(file,0);
    ui->github_release_files->selectRow(file);
    on_github_release_files_itemSelectionChanged();
    if(cmd[1] == "open")
        return on_github_open_file_clicked(),true;
    if(cmd[1] == "download" && cmd.size() == 6)
    {
        ui->download_dir->setText(QString::fromUtf8(cmd[5]));
        ui->download_overwrite->setChecked(false);
        return on_github_download_clicked(),true;
    }
    return fail(usage);
}
void FiberDataHub::on_github_repo_currentIndexChanged(int index)
{
    if(ui->github_repo->currentIndex() < 0 || !fetch_github)
        return;
    QString repo = ui->github_repo->currentData().toString();
    if(tags.find(repo) == tags.end())
    {
        QDir().mkpath(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + "/fiber_data_hub");
        QFile f(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) +
                "/fiber_data_hub/" + QString(repo).replace('/','_') + ".json");

        if(f.open(QFile::ReadOnly))
        {
            auto root = QJsonDocument::fromJson(f.readAll()).object();
            dates[repo] = root["date"].toString();
            tags[repo] = root["tags"].toArray();
        }
        else
        {
            tags[repo] = QJsonArray();
            dates[repo] = QString();
            on_load_tags_clicked();
            return;
        }
    }

    notes.clear();
    assets.clear();
    ui->github_tags->setSortingEnabled(false);
    ui->github_tags->setRowCount(0);

    std::map<QString, std::pair<QString,std::set<std::string> > > agg;
    foreach (const QJsonValue& release, tags[repo])
    {
        auto object = release.toObject();
        auto tag = object.value("tag_name").toString();
        if(tag.length() > 2 && tag[tag.length()-2] == '_' && tag[tag.length()-1] >= '1' && tag[tag.length()-1] <= '9')
            tag.chop(2);

        notes[tag] = object.value("body").toString();
        auto& agg_at_tag = agg[tag];
        agg_at_tag.first = object.value("name").toString();
        auto& names = agg_at_tag.second;
        foreach (const auto& each, object.value("assets").toArray())
        {
            assets[tag].append(each);
            auto fn = each.toObject().value("name").toString().toStdString();
            if (fn.empty() || fn.back()!='z' || tipl::ends_with(fn,{".db.fz",".dz"}))
                continue;
            names.insert(fn.substr(0, std::min(fn.find('_'), fn.find('.'))));
        }
    }
    if(dates[repo].isEmpty())
        ui->tag_date->setText("Loading...");
    else
        ui->tag_date->setText("Last sync:" + dates[repo]);

    for (const auto& each : agg)
    {
        int row=ui->github_tags->rowCount();
        ui->github_tags->insertRow(row);
        ui->github_tags->setItem(row,0,new QTableWidgetItem(each.first));
        ui->github_tags->setItem(row,1,new QTableWidgetItem(QString::number(each.second.second.size())));
        ui->github_tags->setItem(row,2,new QTableWidgetItem(QString::number(assets[each.first].size())));
        ui->github_tags->setItem(row,3,new QTableWidgetItem(each.second.first));
    }

    ui->github_tags->sortByColumn(0,Qt::AscendingOrder);
    ui->github_tags->setSortingEnabled(true);
    ui->github_tags->resizeRowsToContents();
    ui->github_tags->resizeColumnToContents(0);
    ui->github_tags->resizeColumnToContents(1);
    ui->github_tags->resizeColumnToContents(2);
}


void FiberDataHub::on_load_tags_clicked()
{
    if(ui->github_repo->currentIndex() < 0 || !fetch_github)
        return;
    QString repo = ui->github_repo->currentData().toString();
    QString url = QString("https://api.github.com/repos/%1/releases").arg(repo);
    ui->github_tags->setSortingEnabled(false);
    ui->github_tags->setRowCount(0);
    ui->tag_date->setText("Loading...");
    ui->load_tags->setEnabled(false);
    notes.clear();
    assets.clear();
    std::vector<int> per_page = {64,32,16,8,4};
    QTimer::singleShot(0,this, [=](){loadTags(QUrl(url), repo, QJsonArray(), per_page[std::min<int>(per_page.size()-1,github_api_rate_limit/15)]);});
}

QString showQNetworkReplyError(QNetworkReply* reply)
{
    int http_error = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    if(http_error)
        return QMap<int, QString>({
            {301, "Moved Permanently - The requested resource has been permanently moved to a new location."},
            {302, "Found - The requested resource resides temporarily under a different URI."},
            {304, "Not Modified - The server has fulfilled the request, but the document has not been modified."},
            {400, "Bad Request - The request was invalid."},
            {401, "Unauthorized - Valid authentication credentials are required."},
            {404, "Permission Needed - The resource requires access permission."},
            {405, "Method Not Allowed - The request method is not supported for the requested resource."},
            {408, "Request Timeout - The server timed out waiting for the request."},
            {500, "Internal Server Error - The server encountered an unexpected condition."},
            {502, "Bad Gateway - The server received an invalid response from an upstream server."},
            {503, "Service Unavailable - The server is currently unable to handle the request."},
            {504, "Gateway Timeout - The server did not receive a timely response from an upstream server."},
                                      }).value(http_error,"error code: " + QString::number(http_error));

    return reply->errorString();
}

void FiberDataHub::update_rate_limit(QSharedPointer<QNetworkReply> reply)
{
    if(reply->rawHeader("X-RateLimit-Remaining").toInt() == 0)
        return;
    tipl::out() << "api rate limit: " << (github_api_rate_limit = reply->rawHeader("X-RateLimit-Remaining").toInt());
}
void FiberDataHub::loadTags(QUrl url,QString repo,QJsonArray array,int per_page)
{
    static int retryCount = 0;
    {
        QUrlQuery q(url.query());
        q.removeAllQueryItems("per_page");
        q.addQueryItem("per_page", repo.contains("restricted") ? "64" : QString::number(per_page).toStdString().c_str());
        url.setQuery(q);
    }

    tags[repo] = array;
    if (!array.isEmpty() && repo == ui->github_repo->currentData().toString())
        QTimer::singleShot(0, this, [this]() {on_github_repo_currentIndexChanged(0);});

    tipl::out() << "loading " << url.toString().toStdString();
    auto reply = main_window.get(url);
    connect(reply.get(), &QNetworkReply::finished, this, [=]() mutable {
        if (reply->error() != QNetworkReply::NoError)
        {
            if (reply->error() != QNetworkReply::OperationCanceledError) {
                int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
                if (status!=401 && status!=404 && status!=403 && retryCount<5) {
                    int waitTime = 2 << retryCount;  // 2,4,8,16,32s
                    QTimer::singleShot(waitTime*1000, this, [=]() {
                        ++retryCount;
                        loadTags(url, repo, array,per_page);
                    });
                } else {
                    QMessageBox::critical(this, "ERROR", showQNetworkReplyError(reply.get()));
                }
            }
        }
        else
        {
            update_rate_limit(reply);
            retryCount = 0;
            foreach (const QJsonValue& release , QJsonDocument::fromJson(QString(reply->readAll()).toUtf8()).array())
                array.append(release);

            // next page?
            auto m = QRegularExpression("<([^>]+)>; rel=\"next\"").match(reply->rawHeader("Link"));
            if (m.hasMatch())
            {
                QUrl nextPg = m.captured(1);
                if (nextPg.isValid())
                {
                    int delay_time = 0;
                    if(github_api_rate_limit < 40)
                        delay_time = 1000;
                    if(github_api_rate_limit < 20)
                        delay_time = 5000;
                    QTimer::singleShot(delay_time, this, [=]() {loadTags(nextPg, repo, array , per_page);});
                    return;
                }
            }
            tags[repo] = array;
            dates[repo] = QDate::currentDate().toString("yyyy/MM/dd");

            {
                tipl::out() << "saving file list of " << repo.toStdString();
                QDir().mkpath(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + "/fiber_data_hub");

                QJsonObject root;
                root["date"] = dates[repo];
                root["tags"] = tags[repo];

                QFile f(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) +
                        "/fiber_data_hub/" + QString(repo).replace('/','_') + ".json");
                if(f.open(QFile::WriteOnly))
                    f.write(QJsonDocument(root).toJson(QJsonDocument::Compact));
            }

            if (!array.isEmpty() && repo == ui->github_repo->currentData().toString())
                QTimer::singleShot(0, this, [this]() {on_github_repo_currentIndexChanged(0);});
        }
        ui->load_tags->setEnabled(true);
        reply->deleteLater();
    });
}


void FiberDataHub::loadFiles()
{
    bool is_restricted = ui->github_repo->currentText().contains("restricted");
    ui->github_release_files->setSortingEnabled(false);
    ui->github_release_files->setUpdatesEnabled(false);
    ui->github_release_files->setRowCount(0);


    for(int tab = ui->github_release_note->count()-1;tab > 0;--tab)
        ui->github_release_note->removeTab(tab);
    github_tsv_link.resize(1);

    QStringList units = {" b", " kb", " mb", " gb"};
    foreach (const QJsonValue& asset,assets[cur_tag])
    {
        QJsonObject assetObject = asset.toObject();
        size_t size = assetObject.value("size").toInteger();
        int i = 0;
        while (size >= 1024 && i < units.size() - 1)
        {
            size /= 1024;
            i++;
        }
        int row = ui->github_release_files->rowCount();
        auto file_name = assetObject.value("name").toString();
        ui->github_release_files->insertRow(row);
        ui->github_release_files->setItem(row, 0, new QTableWidgetItem(file_name));
        ui->github_release_files->setItem(row, 1, new QTableWidgetItem(QString::number(size)+units[i]));
        ui->github_release_files->setItem(row, 2, new QTableWidgetItem(assetObject.value("created_at").toString()));
        ui->github_release_files->setItem(row, 3, new QTableWidgetItem(QString::number(assetObject.value("download_count").toInteger())));
        if(is_restricted)
            ui->github_release_files->setItem(row, 4, new QTableWidgetItem(assetObject.value("url").toString()));
        else
            ui->github_release_files->setItem(row, 4, new QTableWidgetItem(assetObject.value("browser_download_url").toString()));
        ui->github_release_files->item(row,1)->setData(Qt::UserRole, assetObject.value("size").toInteger()); // Save the original size
        if(file_name.contains(".tsv"))
        {
            ui->github_release_note->addTab(new QWidget(ui->github_release_note),file_name.remove(".tsv"));
            github_tsv_link.push_back(assetObject.value("browser_download_url").toString());
        }
    }
    ui->github_release_files->sortByColumn(0,Qt::AscendingOrder);
    ui->github_release_files->setUpdatesEnabled(true);
    ui->github_release_files->resizeColumnToContents(0);
    ui->github_release_files->resizeColumnToContents(1);
    ui->github_release_files->resizeColumnToContents(2);
    ui->github_release_files->setColumnWidth(3,50);
    ui->github_release_files->setSortingEnabled(true);

    ui->file_count->setText(QString("%1 files").arg(ui->github_release_files->rowCount()));

}


void FiberDataHub::on_github_release_note_currentChanged(int index)
{
    if(index && index < github_tsv_link.size())
    {

        if(!github_tsv_link[index].isEmpty())
        {
            tipl::out() << "downloading " << github_tsv_link[index].toStdString().c_str();
            auto reply = main_window.get(github_tsv_link[index]);
            QEventLoop loop;
            QObject::connect(reply.get(), &QNetworkReply::finished, this, [&loop, this, reply, index]()
            {
                loop.quit();
                if (reply->error() == QNetworkReply::NoError &&
                    index < github_tsv_link.size() &&
                    !github_tsv_link[index].isEmpty())
                {
                    github_tsv_link[index].clear();
                    auto tableWidget = new QTableWidget(ui->github_release_note->widget(index));
                    auto layout = new QVBoxLayout(ui->github_release_note->widget(index));
                    layout->addWidget(tableWidget);

                    QString data = reply->readAll();
                    QStringList rows = data.split("\n");
                    while(rows.count() && rows.back().isEmpty())
                        rows.pop_back();
                    QStringList headers = rows.takeFirst().split("\t");
                    tableWidget->setRowCount(rows.size());
                    tableWidget->setColumnCount(headers.size());
                    tableWidget->setHorizontalHeaderLabels(headers);

                    for (int i = 0; i < rows.size(); ++i) {
                        QStringList cols = rows.at(i).split("\t");
                        for (int j = 0; j < cols.size(); ++j) {
                            QTableWidgetItem* item = new QTableWidgetItem;
                            bool ok;
                            double val = cols.at(j).toDouble(&ok);
                            if (ok)
                                item->setData(Qt::DisplayRole, val);
                            else
                                item->setText(cols.at(j));
                            tableWidget->setItem(i, j, item);
                        }
                    }
                    tableWidget->setSortingEnabled(true);
                }
            });
            loop.exec();

        }
    }
}


void FiberDataHub::on_github_tags_itemSelectionChanged()
{
    if(ui->github_tags->currentRow() >= 0 && ui->github_tags->rowCount())
    {
        cur_tag = ui->github_tags->item(ui->github_tags->currentRow(), 0)->text();
        QString title = ui->github_tags->item(ui->github_tags->currentRow(), 3)->text();
        ui->github_repo_title->setText(title);
        auto content = notes[cur_tag].split('\n');
        if(!content.empty() && content[0].contains(title))
            content.remove(0);
        ui->github_note->setMarkdown(content.join('\n'));
        ui->github_release_note->setCurrentIndex(0);
        loadFiles();
    }
    ui->github_tags->setColumnWidth(3,50);
}

void FiberDataHub::on_browseDownloadDir_clicked()
{
    QString filename =
        QFileDialog::getExistingDirectory(this,"Browse Download Directory",
                                          ui->download_dir->text());
    if ( filename.isEmpty() )
        return;
    ui->download_dir->setText(filename);
}


void FiberDataHub::on_github_release_files_itemSelectionChanged()
{
    int selectedRows = ui->github_release_files->selectionModel()->selectedRows().size();
    ui->github_release_files->setColumnWidth(3,50);
    ui->github_download->setEnabled(selectedRows > 0);
    if(selectedRows == 1 && ui->github_release_files->currentRow() >= 0)
    {
        auto file_name = ui->github_release_files->item(ui->github_release_files->currentRow(),0)->text();
        ui->github_open_file->setText(QString("Open %1").arg(file_name));
        ui->github_open_file->setVisible(true);
        ui->github_open_file_mode->setVisible(true);
        ui->github_open_file_mode->clear();
        ui->github_open_file_mode->addItem("O1: View Image");
        ui->github_open_file_mode->addItem(file_name.endsWith(".src.gz") || file_name.endsWith(".sz") ? "T2: Reconstruction": "T3: Fiber Tracking");
        ui->github_open_file_mode->setCurrentIndex(file_name.endsWith(".nii.gz") || file_name.endsWith(".nii") ? 0 : 1 );
        if(file_name.endsWith(".db.fz") || file_name.endsWith(".db.fib.gz") || file_name.endsWith(".dz"))
        {
            ui->github_open_file_mode->addItems({"C2: View Database","C3: Correlational Tracking"});
            ui->github_open_file_mode->setCurrentIndex(2);
        }
    }
    else
    {
        ui->github_open_file->setVisible(false);
        ui->github_open_file_mode->setVisible(false);
    }
    ui->github_download->setText(selectedRows > 0 ? QString("Download %1 File(s)...").arg(selectedRows) : QString("Download"));
    ui->file_count->setText(QString("%1/%2 files").arg(selectedRows).arg(ui->github_release_files->rowCount()));
}


void FiberDataHub::on_github_select_all_clicked()
{
    ui->github_release_files->selectAll();
}


void FiberDataHub::on_github_download_clicked()
{
    QList<QTableWidgetSelectionRange> ranges = ui->github_release_files->selectedRanges();
    if (ranges.isEmpty()){
        QMessageBox::critical(this, "ERROR", "No files selected for download");
        return;
    }

    std::vector<int> row_list;
    for (int i = 0; i < ranges.size();++i)
        for (int row = ranges[i].topRow(); row <= ranges[i].bottomRow(); ++row)
            row_list.push_back(row);

    tipl::progress p("downloading...",true);
    for (int i = 0; p(i,row_list.size());++i)
    {
        qint64 startTime = QDateTime::currentMSecsSinceEpoch();

        QString url = ui->github_release_files->item(row_list[i], 4)->text();
        QString filePath = ui->download_dir->text() + "/" + ui->github_release_files->item(row_list[i], 0)->text();
        if (QFile::exists(filePath) && !ui->download_overwrite->isChecked())
        {
            tipl::out() << filePath.toStdString() << " exists...skipping";
            continue;
        }

        tipl::out() << url.toStdString();

        QSharedPointer<QNetworkReply> reply;
        int retry = 0;
        const int max_retry = 5;
        while (retry < max_retry)
        {
            reply = main_window.get(url);
            qint64 bytesTotal = ui->github_release_files->item(row_list[i], 1)->data(Qt::UserRole).toLongLong();
            while (!reply->isFinished() && !p.aborted())
            {
                QCoreApplication::processEvents();
                QThread::msleep(100); // Check every 100ms
            }
            if (reply->error() == QNetworkReply::NoError)
                break;
            retry++;
            QThread::sleep(3);
        }

        if (retry >= max_retry)
        {
            QMessageBox::critical(this, "ERROR", showQNetworkReplyError(reply.get()));
            return;
        }
        if (p.aborted())
            return;

        {
            auto file = std::make_shared<QFile>(filePath);
            if (!file->open(QFile::WriteOnly))
            {
                QMessageBox::critical(this, "ERROR", "Failed to save file to disk");
                return;
            }
            QTimer::singleShot(0, this, [file, reply]()
            {
                file->write(reply->readAll());
            });
        }
    }
}


void FiberDataHub::on_github_select_matching_clicked()
{
    tipl::progress p("select matching");
    QString pattern = QInputDialog::getText(this, "Select Matching", "Enter a sub text (fib.gz), wild card (*.fib.gz) or regex pattern:");
    if (pattern.isEmpty())
        return;
    Qt::MatchFlag flags = Qt::MatchContains;
    if(pattern.contains("*"))
        flags = Qt::MatchWildcard;
    else
    if(pattern.contains(QRegularExpression("[.^$|()\\[\\]{}*+?\\\\]")))
    {
        QRegularExpression regex(pattern);
        if (regex.isValid())
            flags = Qt::MatchRegularExpression;
        else
        {
            QMessageBox::critical(this,"ERROR","Invalid regular expression pattern");
            return;
        }
    }
    QList<QTableWidgetItem*> items = ui->github_release_files->findItems(pattern, flags);
    ui->github_release_files->blockSignals(true);
    ui->github_release_files->clearSelection();
    for (int i = 0; p(i, items.size()); ++i)
        ui->github_release_files->setRangeSelected(QTableWidgetSelectionRange(items[i]->row(), 0, items[i]->row(), ui->github_release_files->columnCount() - 1), true);
    ui->github_release_files->blockSignals(false);
    on_github_release_files_itemSelectionChanged();
}





void FiberDataHub::on_github_open_file_clicked()
{
    auto row = ui->github_release_files->currentRow();
    if(row < 0)
        return;
    QDir dir(QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/" + cur_tag);
    if (!dir.exists() && !dir.mkpath("."))
        return QMessageBox::critical(this,"ERROR","cannot create a temporary directory to store file"),void();

    QString filePath = dir.path()+ "/" + ui->github_release_files->item(row, 0)->text();
    auto git_open = [this,filePath](void)
    {
        if(filePath.endsWith(".nii.gz") || filePath.endsWith(".nii") ||
           filePath.endsWith(".fib.gz") || filePath.endsWith(".fz") || filePath.endsWith(".dz"))
        {

            if(ui->github_open_file_mode->currentIndex() == 0)
                main_window.loadNii(QStringList() << filePath);
            else
            if(ui->github_open_file_mode->currentIndex() == 1)
                main_window.loadFib(filePath);
            else
            if(ui->github_open_file_mode->currentIndex() > 1) // open db
            {
                auto database = std::make_shared<group_connectometry_analysis>();
                tipl::progress prog("reading connectometry db");
                if(!database->load_database(filePath.toStdString().c_str()))
                {
                    QMessageBox::critical(this,"ERROR",database->error_msg.c_str());
                    return;
                }
                if(ui->github_open_file_mode->currentIndex() == 2)
                {
                    auto db = new db_window(&main_window,database);
                    db->setWindowTitle(filePath);
                    db->setAttribute(Qt::WA_DeleteOnClose);
                    db->show();
                }
                else
                {
                    auto group_cnt = new group_connectometry(&main_window,database,filePath);
                    group_cnt->setAttribute(Qt::WA_DeleteOnClose);
                    group_cnt->show();
                }
            }
        }
        else
        {
            if(ui->github_open_file_mode->currentIndex() == 0)
                main_window.loadNii(QStringList() << filePath);
            else
                main_window.openFile(QStringList() << filePath);
        }
    };


    qint64 bytesTotal = ui->github_release_files->item(row, 1)->data(Qt::UserRole).toLongLong();
    if (QFile::exists(filePath) && !ui->download_overwrite->isChecked())
    {
        git_open();
        return;
    }
    tipl::out() << "download file to " << filePath.toStdString();
    auto reply = main_window.get(ui->github_release_files->item(row, 4)->text());

    // Create a progress dialog
    QProgressDialog progressDialog("Downloading...", "Cancel", 0, 100, this);
    progressDialog.setModal(true);
    progressDialog.show();
    qint64 bytesReceived = 0;
    QEventLoop loop;

    QObject::connect(reply.get(), &QNetworkReply::readyRead, this,
                     [this, &progressDialog, &bytesReceived, bytesTotal,reply]()
    {
        progressDialog.setValue((reply->bytesAvailable() * 100) / (bytesTotal));
    });
    QObject::connect(reply.get(), &QNetworkReply::finished, this,
                     [this, filePath, git_open, &progressDialog, &loop,reply]() // Pass the loop to the lambda
    {
        if (reply->error() != QNetworkReply::NoError)
        {
            if(reply->error() != QNetworkReply::OperationCanceledError)
                QMessageBox::critical(this, "ERROR", showQNetworkReplyError(reply.get()));
        }
        else
        {
            auto downloadFile = std::make_shared<QFile>(filePath);
            if (!downloadFile->open(QFile::WriteOnly))
            {
                QMessageBox::critical(this, "ERROR", "Failed to open file for writing");
                return;
            }
            downloadFile->write(reply->readAll());
            downloadFile->close();
            QTimer::singleShot(0, this, [git_open](){git_open();});
        }
        progressDialog.close();
        loop.quit();
    });

    QObject::connect(&progressDialog, &QProgressDialog::canceled, this, [this,&loop,reply]() // Pass the loop to the lambda
    {
        if (reply && reply->isRunning())
            reply->abort();
    });

    loop.exec();
}

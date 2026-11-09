#ifndef FIBER_DATA_HUB_HPP
#define FIBER_DATA_HUB_HPP

#include <QBrush>
#include <QFile>
#include <QJsonArray>
#include <QLinearGradient>
#include <QMainWindow>
#include <QNetworkReply>
#include <QSettings>
#include <QSharedPointer>
#include <QTableWidget>
#include <QTimer>
#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "TIPL/tipl.hpp"

QString showQNetworkReplyError(QNetworkReply* reply);

// streams a reply (owned by the caller) to file_name chunk by chunk without blocking; done(ok,error) is called
// once at the end, with an empty error for a cancel (reply->abort()), and the partial file is removed unless ok
inline void download_to_file(QNetworkReply* reply,const QString& file_name,
                             std::function<void(qint64,qint64)> progress,std::function<void(bool,QString)> done)
{
    auto file = QSharedPointer<QFile>::create(file_name);
    if(!file->open(QFile::WriteOnly))
        return reply->abort(),done(false,"Failed to open file for writing");
    auto write = [=](void)
    {
        auto data = reply->readAll();
        if(file->write(data) != data.size())
            reply->abort();
    };
    QObject::connect(reply,&QNetworkReply::readyRead,reply,write);
    QObject::connect(reply,&QNetworkReply::downloadProgress,reply,progress);
    QObject::connect(reply,&QNetworkReply::finished,reply,[=](void)
    {
        write();
        bool write_failed = file->error() != QFile::NoError; // checked before close() resets it
        file->close();
        if(!write_failed && reply->error() == QNetworkReply::NoError)
            return done(true,QString());
        file->remove();
        done(false,write_failed ? QString("Failed to save file to disk") :
                   reply->error() == QNetworkReply::OperationCanceledError ? QString() : showQNetworkReplyError(reply));
    });
}

// downloads files 3 at a time with up to 5 tries each. a group (e.g. a repo) is canceled as a whole,
// a key (e.g. a table row) may cover several files, and finished files keep their bars until clear_finished()
struct download_queue
{
    struct item
    {
        QString group,key,url,file_name;
        float progress = 0.0f; // downloaded fraction, -1 for a failed file
        bool done = false;
        int retry = 0;
        QSharedPointer<QNetworkReply> reply; // null while queued
    };
    std::vector<item> items;
    QString error;
    bool running = false;
    QObject context; // owns the retry timers, so none fires after destruction

    std::function<QSharedPointer<QNetworkReply>(const QString& url)> get;
    std::function<void(const QString& key)> on_progress = [](const QString&){}; // an empty key means all keys
    std::function<void(void)> on_change = [](void){};
    std::function<void(const QString& error)> on_finished = [](const QString&){};

    ~download_queue(void)
    {
        on_progress = [](const QString&){}; // the owner's table may be gone
        for(auto each : std::vector<item>(items)) // abort removes partial files and calls back into items
            if(each.reply)
                each.reply->abort();
    }
    auto find(const QString& file_name)
    {
        return std::find_if(items.begin(),items.end(),[&](const auto& each){return each.file_name == file_name;});
    }
    bool busy(const QString& group) const
    {
        return std::any_of(items.begin(),items.end(),[&](const auto& each){return each.group == group && !each.done;});
    }
    QBrush brush(const QString& key) const
    {
        // fill from the left: blue for the downloaded fraction, red if any file failed
        float sum = 0.0f;
        size_t count = 0;
        bool failed = false;
        for(const auto& each : items)
            if(each.key == key)
                sum += each.progress,++count,failed = failed || each.progress < 0.0f;
        if(!count)
            return QBrush();
        float x = failed ? 1.0f : sum/float(count);
        QLinearGradient gradient(0,0,1,0);
        gradient.setCoordinateMode(QGradient::ObjectBoundingMode);
        gradient.setColorAt(x,failed ? QColor(244,160,160) : QColor(159,208,245));
        if(x < 1.0f)
            gradient.setColorAt(std::min(x+0.001f,1.0f),Qt::transparent);
        return gradient;
    }
    // paints the bar column of every row whose key column matches key (all rows for an empty key)
    void paint(QTableWidget* table,int key_column,int bar_column,const QString& key) const
    {
        for(int row = 0;row < table->rowCount();++row)
            if(auto row_key = table->item(row,key_column)->text();key.isEmpty() || row_key == key)
                table->item(row,bar_column)->setBackground(brush(row_key));
    }
    void add(const QString& group,const QString& url,const QString& file_name,bool overwrite = false,QString key = QString())
    {
        if(find(file_name) != items.end())
            return;
        if(key.isEmpty())
            key = url;
        items.push_back({group,key,url,file_name});
        if(!overwrite && QFile::exists(file_name))
        {
            tipl::out() << file_name.toStdString() << " exists...skipping";
            items.back().progress = 1.0f;
            items.back().done = true;
        }
        on_progress(key);
    }
    void clear_finished(void)
    {
        items.erase(std::remove_if(items.begin(),items.end(),[](const auto& each){return each.done;}),items.end());
        on_progress(QString());
    }
    void cancel(const QString& group)
    {
        for(auto each : std::vector<item>(items)) // aborting calls back into items
            if(each.group == group && each.reply)
                each.reply->abort();
        items.erase(std::remove_if(items.begin(),items.end(),[&](const auto& each){return each.group == group && !each.done;}),items.end());
        if(std::all_of(items.begin(),items.end(),[](const auto& each){return each.done;}))
            running = false,error.clear();
        on_progress(QString());
        next();
    }
    void next(void)
    {
        auto it = std::find_if(items.begin(),items.end(),[](const auto& each){return !each.reply && !each.done;});
        if(it == items.end() ||
           std::count_if(items.begin(),items.end(),[](const auto& each){return bool(each.reply);}) >= 3) // 3 parallel downloads
        {
            if((running || !error.isEmpty()) && std::all_of(items.begin(),items.end(),[](const auto& each){return each.done;}))
                running = false,on_finished(std::exchange(error,QString()));
            return on_change();
        }

        running = true;
        QString file_name = it->file_name;
        tipl::out() << it->url.toStdString();
        it->reply = get(it->url);
        download_to_file(it->reply.get(),file_name,[this,file_name](qint64 received,qint64 total)
        {
            auto each = find(file_name);
            if(total > 0 && int(received*100/total) != int(each->progress*100))
                each->progress = float(received)/float(total),on_progress(each->key);
        },[this,file_name](bool ok,QString reply_error)
        {
            auto each = find(file_name);
            bool failed = !ok && !reply_error.isEmpty(),retry = failed && ++each->retry < 5;
            if(retry)
                tipl::warning() << reply_error.toStdString() << ", retrying " << each->url.toStdString();
            if(failed && !retry)
                error = reply_error;
            each->reply.reset(); // back to the queue unless done
            each->done = ok || (failed && !retry);
            each->progress = ok ? 1.0f : (each->done ? -1.0f : 0.0f);
            on_progress(each->key);
            QTimer::singleShot(retry ? 3000 : 0,&context,[this](void){next();});
        });
        next(); // start more until the parallel limit
    }
};


class MainWindow;
class QUrl;

namespace Ui {
class FiberDataHub;
}

class FiberDataHub : public QMainWindow
{
    Q_OBJECT
    MainWindow& main_window;
    Ui::FiberDataHub* ui;
    QSettings settings;
    std::vector<QString> github_tsv_link;
    int github_api_rate_limit = 60;
    QString cur_tag;
    bool fetch_github = false;
    std::map<QString,QString> notes,dates;
    std::map<QString,QJsonArray> tags,assets;
    bool initialize(void);
    void loadTags(QUrl,QString,QJsonArray,int);
    void loadFiles(void);
    void update_rate_limit(QSharedPointer<QNetworkReply>);

    download_queue downloads; // grouped by repo, keyed by url
    void download_selected(void);

public:
    explicit FiberDataHub(MainWindow* parent);
    ~FiberDataHub();
    std::string error_msg;
    bool command(const std::vector<std::string>& cmd);
    void list_restricted_repos(void);

private slots:
    void on_load_tags_clicked();
    void on_github_tags_itemSelectionChanged();
    void on_browseDownloadDir_clicked();
    void on_github_release_files_itemSelectionChanged();
    void on_github_select_all_clicked();
    void on_github_download_clicked();
    void on_github_select_matching_clicked();
    void on_github_release_note_currentChanged(int index);
    void on_github_repo_currentIndexChanged(int index);
    void on_github_open_file_clicked();
};

#endif

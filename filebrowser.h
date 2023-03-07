#ifndef FILEBROWSER_H
#define FILEBROWSER_H
#include <memory>
#include <QMainWindow>
#include <QGraphicsScene>
#include <QTableWidget>
#include "zlib.h"
#include "TIPL/tipl.hpp"


namespace Ui {
    class FileBrowser;
}
class FileBrowser : public QMainWindow
{
    Q_OBJECT
    QGraphicsScene scene;
public:
    unsigned int cur_z;
    tipl::image<3> data;
    tipl::color_image slice_image;
    QImage view_image;
public:
    explicit FileBrowser(QWidget *parent);
    ~FileBrowser();
    QString image_file_name;
    std::map<std::string,std::string> mon_map;
private:
    Ui::FileBrowser *ui;
    QTimer* timer;
    QStringList image_list;
    std::vector<std::vector<float> > b_value_list;
    std::vector<std::vector<float> > b_vec_list;
private:
    QString preview_file_name;
    float preview_voxel_size[3];
    bool preview_loaded;
    tipl::image<3> preview_data;
    std::shared_ptr<std::future<void>> preview_thread;
    void preview_image(QString file_name);
private slots:

    void on_tableWidget_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn);
    void on_changeWorkDir_clicked();
    void on_subject_list_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn);
    void on_refresh_list_clicked();
    void on_create_src_clicked();

public slots:
    void show_image();
    void populateDirs();
};

#endif // FILEBROWSER_H

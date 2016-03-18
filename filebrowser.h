#ifndef FILEBROWSER_H
#define FILEBROWSER_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <memory>
#include <image/image.hpp>
#include <QTableWidget>


namespace Ui {
    class FileBrowser;
}
class FileBrowser : public QMainWindow
{
    Q_OBJECT
    QGraphicsScene scene;
public:
    unsigned int cur_z;
    image::basic_image<float,3> data;
    image::color_image slice_image;
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
    image::basic_image<float,3> preview_data;
    std::auto_ptr<std::future<void>> preview_thread;
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

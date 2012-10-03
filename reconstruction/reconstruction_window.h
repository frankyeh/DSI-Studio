#ifndef RECONSTRUCTION_WINDOW_H
#define RECONSTRUCTION_WINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QSettings>
#include <image/image.hpp>

namespace Ui {
    class reconstruction_window;
}

class ImageModel;
class reconstruction_window : public QMainWindow
{
    Q_OBJECT
    QSettings settings;
public:
    QString absolute_path;
    QStringList filenames;
    explicit reconstruction_window(QStringList filenames_,QWidget *parent = 0);
    ~reconstruction_window();
protected:
    void resizeEvent ( QResizeEvent * event );
    void showEvent ( QShowEvent * event );
    void closeEvent(QCloseEvent *event);
private:
    Ui::reconstruction_window *ui;
    QGraphicsScene scene;
    image::basic_image<image::rgb_color> buffer;
    QImage slice_image;
    ImageModel* handle;
    float params[5];
    image::basic_image<unsigned char, 3>image;
    image::basic_image<unsigned char, 3>mask;
    void load_src(int index);
    void doReconstruction(unsigned char method_id,bool prompt);
private slots:
    void on_QDif_toggled(bool checked);
    void on_GQI_toggled(bool checked);
    void on_QBI_toggled(bool checked);
    void on_DSI_toggled(bool checked);
    void on_DTI_toggled(bool checked);

    void on_load_mask_clicked();
    void on_save_mask_clicked();
    void on_thresholding_clicked();
    void on_doDTI_clicked();
    void on_smoothing_clicked();
    void on_defragment_clicked();
    void on_dilation_clicked();
    void on_erosion_clicked();
    void on_SlicePos_sliderMoved(int position);

    void on_QSDRT_toggled(bool checked);
    void on_ODFSharpening_currentIndexChanged(int index);
    void on_Decomposition_currentIndexChanged(int index);
    void on_remove_background_clicked();
};

#endif // RECONSTRUCTION_WINDOW_H

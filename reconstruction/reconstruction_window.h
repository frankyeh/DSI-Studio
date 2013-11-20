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
    QGraphicsScene source;
    image::color_image buffer_source;
    QImage source_image;
    float max_source_value,source_ratio;

    void load_b_table(void);
private:
    QGraphicsScene scene;
    image::color_image buffer;
    QImage slice_image;
private:
    Ui::reconstruction_window *ui;
    image::geometry<3> dim;

    std::auto_ptr<ImageModel> handle;
    float params[6];
    image::basic_image<unsigned char, 3>image;
    image::basic_image<unsigned char, 3>mask;
    bool load_src(int index);
    void doReconstruction(unsigned char method_id,bool prompt);
private slots:
    void on_QDif_toggled(bool checked);
    void on_GQI_toggled(bool checked);
    void on_QBI_toggled(bool checked);
    void on_DSI_toggled(bool checked);
    void on_DTI_toggled(bool checked);
    void on_RDSI_toggled(bool checked);

    void on_load_mask_clicked();
    void on_save_mask_clicked();
    void on_thresholding_clicked();
    void on_doDTI_clicked();
    void on_smoothing_clicked();
    void on_defragment_clicked();
    void on_dilation_clicked();
    void on_erosion_clicked();
    void on_SlicePos_sliderMoved(int position);

    void on_remove_background_clicked();
    void on_b_table_itemSelectionChanged();
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();
    void on_manual_reg_clicked();
    void on_odf_sharpening_currentIndexChanged(int index);
    void on_RFSelection_currentIndexChanged(int index);
    void on_AdvancedOptions_clicked();
    void on_save4dnifti_clicked();
    void on_save_b_table_clicked();
    void on_save_bvals_clicked();
    void on_save_bvec_clicked();
};

#endif // RECONSTRUCTION_WINDOW_H

#ifndef RECONSTRUCTION_WINDOW_H
#define RECONSTRUCTION_WINDOW_H
#include <QMessageBox>
#include <QMainWindow>
#include <QGraphicsScene>
#include <QSettings>
#include <tipl/tipl.hpp>


namespace Ui {
    class reconstruction_window;
}

struct ImageModel;
bool add_other_image(ImageModel* handle,QString name,QString filename,bool full_auto);
class reconstruction_window : public QMainWindow
{
    Q_OBJECT
    QSettings settings;
public:
    QString absolute_path;
    QStringList filenames;
    explicit reconstruction_window(QStringList filenames_,QWidget *parent = 0);
    ~reconstruction_window();
    std::vector<std::string> steps;
    void command(QString cmd);
protected:
    void resizeEvent ( QResizeEvent * event );
    void showEvent ( QShowEvent * event );
    void closeEvent(QCloseEvent *event);
private:
    QGraphicsScene source;
    tipl::color_image buffer_source;
    QImage source_image;
    float max_source_value,source_ratio;
    void load_b_table(void);
private:
    QGraphicsScene scene;
    tipl::color_image buffer;
    QImage slice_image;
    tipl::value_to_color<float> v2c;
    char view_orientation = 2;
private:
    Ui::reconstruction_window *ui;
    std::auto_ptr<ImageModel> handle;
    bool load_src(int index);
    void update_dimension(void);
    void doReconstruction(unsigned char method_id,bool prompt);
private slots:
    void on_QSDR_toggled(bool checked);
    void on_GQI_toggled(bool checked);
    void on_DTI_toggled(bool checked);
    void on_load_mask_clicked();
    void on_save_mask_clicked();
    void on_doDTI_clicked();

    void on_smoothing_clicked(){command("[Step T2a][Smoothing]");}
    void on_defragment_clicked(){command("[Step T2a][Defragment]");}
    void on_dilation_clicked(){command("[Step T2a][Dilation]");}
    void on_erosion_clicked(){command("[Step T2a][Erosion]");}
    void on_negate_clicked(){command("[Step T2a][Negate]");}
    void on_remove_background_clicked(){command("[Step T2a][Remove Background]");}
    void on_thresholding_clicked(){command("[Step T2a][Threshold]");}

    void on_actionFlip_x_triggered(){command("[Step T2][Edit][Image flip x]");}
    void on_actionFlip_y_triggered(){command("[Step T2][Edit][Image flip y]");}
    void on_actionFlip_z_triggered(){command("[Step T2][Edit][Image flip z]");}
    void on_actionFlip_xy_triggered(){command("[Step T2][Edit][Image swap xy]");}
    void on_actionFlip_yz_triggered(){command("[Step T2][Edit][Image swap yz]");}
    void on_actionFlip_xz_triggered(){command("[Step T2][Edit][Image swap xz]");}
    void on_actionRotate_to_MNI_triggered(){command("[Step T2][Edit][Rotate to MNI]");}

    void on_actionTrim_image_triggered(){command("[Step T2][Edit][Trim]");}
    void on_actionFlip_bx_triggered(){command("[Step T2][Edit][Change b-table:flip bx]");QMessageBox::information(this,"DSI Studio","B-table flipped",0);}
    void on_actionFlip_by_triggered(){command("[Step T2][Edit][Change b-table:flip by]");QMessageBox::information(this,"DSI Studio","B-table flipped",0);}
    void on_actionFlip_bz_triggered(){command("[Step T2][Edit][Change b-table:flip bz]");QMessageBox::information(this,"DSI Studio","B-table flipped",0);}

    void on_b_table_itemSelectionChanged();
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();
    void on_AdvancedOptions_clicked();
    void on_actionSave_4D_nifti_triggered();
    void on_actionSave_b_table_triggered();
    void on_actionSave_bvals_triggered();
    void on_actionSave_bvecs_triggered();

    void on_actionRotate_triggered();
    void on_delete_2_clicked();
    void on_SlicePos_valueChanged(int value);
    void on_motion_correction_clicked();
    void on_scheme_balance_toggled(bool checked);
    void on_half_sphere_toggled(bool checked);
    void on_add_t1t2_clicked();
    void on_actionManual_Rotation_triggered();
    void on_actionReplace_b0_by_T2W_image_triggered();
    void on_actionCorrect_AP_PA_scans_triggered();
    void on_actionSave_b0_triggered();
    void on_actionEnable_TEST_features_triggered();
    void on_actionImage_upsample_to_T1W_TESTING_triggered();
    void on_open_ddi_study_src_clicked();
    void on_SagView_clicked();
    void on_CorView_clicked();
    void on_AxiView_clicked();
    void on_actionSave_DWI_sum_triggered();

};

#endif // RECONSTRUCTION_WINDOW_H

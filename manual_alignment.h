#ifndef MANUAL_ALIGNMENT_H
#define MANUAL_ALIGNMENT_H
#include <future>
#include <QDialog>
#include <QTimer>
#include <QGraphicsScene>
#include "fib_data.hpp"
namespace Ui {
class manual_alignment;
}

class fib_data;

class manual_alignment : public QDialog
{
    Q_OBJECT
private:
    tipl::image<3> from_original;
    tipl::image<3> from,to,warped_from;
    tipl::image<3> from2,to2,warped_from2;
    tipl::vector<3> from_vs,to_vs;
    QGraphicsScene scene[3];
    float from_downsample = 1.0f;
    float to_downsample = 1.0f;
    tipl::thread thread;
    tipl::transformation_matrix<float> T,iT;
    void load_param(void);
public:
    std::thread warp_image_thread;
    bool free_thread = false;
    bool image_need_update = false;
    bool warp_image_ready = false;
    void warp_image();
public:
    tipl::affine_transform<float> arg;
    std::vector<tipl::image<3> > other_images;
    std::vector<std::string> other_images_name;
    tipl::matrix<4,4> nifti_srow = tipl::identity_matrix();
    std::vector<tipl::transformation_matrix<float> > other_image_T;
public:
    QTimer* timer;
    explicit manual_alignment(QWidget *parent,
                              tipl::image<3> from_,
                              tipl::image<3> from2_,
                              const tipl::vector<3>& from_vs,
                              tipl::image<3> to_,
                              tipl::image<3> to2_,
                              const tipl::vector<3>& to_vs,
                              tipl::reg::reg_type reg_type,
                              tipl::reg::cost_type cost_function);

    ~manual_alignment();
    void connect_arg_update();
    void disconnect_arg_update();
    void add_image(const std::string& name,tipl::image<3> new_image,const tipl::transformation_matrix<float>& T)
    {
        other_images_name.push_back(name);
        other_images.push_back(std::move(new_image));
        other_image_T.push_back(T);
    }
    void add_images(std::shared_ptr<fib_data> handle);
    tipl::transformation_matrix<float> get_iT(void);
private slots:
    void slice_pos_moved();
    void param_changed();
    void on_buttonBox_accepted();

    void on_buttonBox_rejected();

    void on_switch_view_clicked();

    void on_actionSave_Warped_Image_triggered();

    void on_advance_options_clicked();

    void on_files_clicked();

    void on_actionSave_Transformation_triggered();

    void on_actionLoad_Transformation_triggered();

    void on_actionApply_Transformation_triggered();

    void on_actionSmooth_Signals_triggered();

    void on_actionSobel_triggered();

    void on_pushButton_clicked();

    void on_refine_clicked();

public slots:
    void on_rerun_clicked();
    void check_reg();
private:
    Ui::manual_alignment *ui;
    void update_image(void);
};

#endif // MANUAL_ALIGNMENT_H

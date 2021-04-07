#ifndef MANUAL_ALIGNMENT_H
#define MANUAL_ALIGNMENT_H
#include <future>
#include <QDialog>
#include <QTimer>
#include <QGraphicsScene>
#include "tipl/tipl.hpp"
#include "fib_data.hpp"
namespace Ui {
class manual_alignment;
}

class fib_data;

class manual_alignment : public QDialog
{
    Q_OBJECT
public:
    tipl::image<float,3> from_original;
    tipl::image<float,3> from,to,warped_from;
    tipl::matrix<4,4,float> nifti_srow;
    tipl::affine_transform<float> arg,b_upper,b_lower;
    tipl::vector<3> from_vs,to_vs;
    QGraphicsScene scene[3];
    tipl::color_image buffer[3];
    QImage slice_image[3];
    tipl::reg::reg_type reg_type;
public:
    std::vector<tipl::image<float,3> > other_images;
    std::vector<std::string> other_images_name;
    std::vector<tipl::transformation_matrix<float> > other_image_T;
private:
    tipl::thread thread;
private:

    void load_param(void);
public:
    tipl::transformation_matrix<double> T,iT;
public:
    QTimer* timer;
    explicit manual_alignment(QWidget *parent,
                              tipl::image<float,3> from_,
                              const tipl::vector<3>& from_vs,
                              tipl::image<float,3> to_,
                              const tipl::vector<3>& to_vs,
                              tipl::reg::reg_type reg_type,
                              tipl::reg::cost_type cost_function);
    ~manual_alignment();
    void connect_arg_update();
    void disconnect_arg_update();
    void add_image(const std::string& name,tipl::image<float,3> new_image,const tipl::transformation_matrix<float>& T)
    {
        other_images_name.push_back(name);
        other_images.push_back(std::move(new_image));
        other_image_T.push_back(T);
    }
    void add_images(std::shared_ptr<fib_data> handle);
private slots:
    void slice_pos_moved();
    void param_changed();
    void on_buttonBox_accepted();

    void on_buttonBox_rejected();

    void on_switch_view_clicked();


    void on_reg_type_currentIndexChanged(int index);

    void on_actionSave_Warpped_Image_triggered();

    void on_advance_options_clicked();

    void on_files_clicked();

public slots:
    void on_rerun_clicked();
    void check_reg();
private:
    Ui::manual_alignment *ui;
    void update_image(void);
};

#endif // MANUAL_ALIGNMENT_H

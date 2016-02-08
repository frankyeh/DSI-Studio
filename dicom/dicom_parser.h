#ifndef DICOM_PARSER_H
#define DICOM_PARSER_H
#include "boost/ptr_container/ptr_vector.hpp"
#include "dwi_header.hpp"

#include <QDialog>

namespace Ui {
    class dicom_parser;
}
struct compare_qstring{
    bool operator()(const QString& lhs,const QString& rhs)
    {
        if(lhs.length() != rhs.length())
            return lhs.length() < rhs.length();
        return lhs < rhs;
    }
};
class dicom_parser : public QDialog
{
    Q_OBJECT

public:
    explicit dicom_parser(QStringList file_list,QWidget *parent = 0);
    ~dicom_parser();
    void set_name(QString name);
    void update_b_table(void);
private:
    Ui::dicom_parser *ui;
    QString cur_path;
    boost::ptr_vector<DwiHeader> dwi_files;
    std::vector<float> slice_orientation;// for applying slice orientation
    void load_files(QStringList file_list);
private slots:
    void on_upperDir_clicked();
    void on_loadImage_clicked();
    void on_pushButton_clicked();
    void on_buttonBox_accepted();
    void on_load_bval_clicked();
    void on_load_bvec_clicked();
    void on_load_b_table_clicked();
    void on_save_b_table_clicked();
    void on_flip_x_clicked();
    void on_flip_y_clicked();
    void on_flip_z_clicked();
    void on_switch_xy_clicked();
    void on_swith_xz_clicked();
    void on_switch_yz_clicked();
    void on_detect_motion_clicked();
};

#endif // DICOM_PARSER_H

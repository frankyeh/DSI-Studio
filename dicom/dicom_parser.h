#ifndef DICOM_PARSER_H
#define DICOM_PARSER_H
#include "boost/ptr_container/ptr_vector.hpp"
#include "dwi_header.hpp"

#include <QDialog>

namespace Ui {
    class dicom_parser;
}

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
    void on_apply_slice_orientation_clicked();
    void on_upperDir_clicked();
    void on_loadImage_clicked();
    void on_pushButton_clicked();
    void on_toolButton_8_clicked();
    void on_toolButton_2_clicked();
    void on_toolButton_7_clicked();
    void on_toolButton_6_clicked();
    void on_toolButton_5_clicked();
    void on_toolButton_4_clicked();
    void on_toolButton_3_clicked();
    void on_toolButton_clicked();
    void on_buttonBox_accepted();
    void on_load_bval_clicked();
    void on_load_bvec_clicked();
    void on_motion_correction_clicked();
};

#endif // DICOM_PARSER_H

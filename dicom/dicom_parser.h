#ifndef DICOM_PARSER_H
#define DICOM_PARSER_H
#include "dwi_header.hpp"

#include <QMainWindow>

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
class dicom_parser : public QMainWindow
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
    std::vector<std::shared_ptr<DwiHeader> > dwi_files;
    std::vector<float> slice_orientation;// for applying slice orientation
    void load_table(void);
    void load_files(QStringList file_list);
private slots:
    void on_upperDir_clicked();
    void on_pushButton_clicked();
    void on_buttonBox_accepted();
    void on_buttonBox_rejected();
    void on_actionOpen_Images_triggered();
    void on_actionOpen_b_table_triggered();
    void on_actionOpen_bval_triggered();
    void on_actionOpen_bvec_triggered();
    void on_actionSave_b_table_triggered();
    void on_actionFlip_bx_triggered();
    void on_actionFlip_by_triggered();
    void on_actionFlip_bz_triggered();
    void on_actionSwap_bx_by_triggered();
    void on_actionSwap_bx_bz_triggered();
    void on_actionSwap_by_bz_triggered();
    void on_actionDetect_Motion_triggered();
};

#endif // DICOM_PARSER_H

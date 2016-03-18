#ifndef MOTION_DIALOG_HPP
#define MOTION_DIALOG_HPP
#include <QDialog>
#include "dwi_header.hpp"
#include "image/image.hpp"

namespace Ui {
class motion_dialog;
}
class dicom_parser;
class motion_dialog : public QDialog
{
    Q_OBJECT
private:
    dicom_parser& dicom_gui;
    std::vector<std::shared_ptr<DwiHeader> >& dwi_files;
    std::vector<unsigned int> b0_index;
    std::vector<image::affine_transform<double> > arg;
    bool terminated;
    unsigned int finished;
    std::vector<std::shared_ptr<std::future<void> > > threads;
    std::auto_ptr<QTimer> timer;
public:
    explicit motion_dialog(QWidget *parent,std::vector<std::shared_ptr<DwiHeader> >& dwi_files_);
    ~motion_dialog();
    
public slots:
    void show_progress(void);
private slots:
    void on_correction_clicked();

private:
    Ui::motion_dialog *ui;
};

#endif // MOTION_DIALOG_HPP

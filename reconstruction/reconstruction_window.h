#ifndef RECONSTRUCTION_WINDOW_H
#define RECONSTRUCTION_WINDOW_H
#include <QMessageBox>
#include <QMainWindow>
#include <QGraphicsScene>
#include <QSettings>
#include "zlib.h"
#include "TIPL/tipl.hpp"


namespace Ui {
    class reconstruction_window;
}

struct src_data;
enum class command_source;
class QPushButton;
struct preproc_step{QPushButton* button;const char* cmd;const char* report;int path,q,when;}; // report: its text in the processing report; path: column in the flow (-1: shared); runs if question q is answered "when" (1: yes, 2: no)
class reconstruction_window : public QMainWindow
{
    Q_OBJECT
    QSettings settings;
public:
    QString absolute_path;
    QStringList filenames;
    explicit reconstruction_window(QStringList filenames_,QWidget *parent = nullptr);
    ~reconstruction_window();
    bool command(std::vector<std::string> cmd,command_source source);
    std::string error_msg;
public:
    std::vector<QCheckBox*> outputs,adv_outputs;
protected:
    void resizeEvent(QResizeEvent * event) override;
    void showEvent(QShowEvent * event) override;
    void closeEvent(QCloseEvent *event) override;
    bool eventFilter(QObject* obj,QEvent* event) override;
private:
    QGraphicsScene source;
    float source_ratio = 1.0f;
    void load_b_table(void);
private:
    QGraphicsScene scene;
    tipl::value_to_color<float> v2c;
    unsigned char view_orientation = 2;
private: //bad slices
    bool bad_slice_analyzed = false;
    std::vector<std::pair<size_t,size_t> > bad_slices;
private:
    Ui::reconstruction_window *ui;
    std::shared_ptr<src_data> handle;
    std::string existing_steps;
    bool load_src(int index);
    void update_dimension(void);
private: // preprocessing flow
    std::string rev_pe_file,t1w_file;
    std::vector<std::pair<QPushButton*,QPushButton*> > pills; // yes/no of each question
    std::vector<preproc_step> preproc; // in running order
    int path = 0; // the chosen column
    void update_preproc(void);
private slots:
    void on_QSDR_toggled(bool checked);
    void on_GQI_toggled(bool checked);
    void on_DTI_toggled(bool checked);
    void on_actionSave_Mask_triggered();

    void on_b_table_itemSelectionChanged();
    void on_zoom_in_clicked();
    void on_zoom_out_clicked();

    void on_actionRotate_triggered();
    void on_delete_2_clicked();
    void on_SlicePos_valueChanged(int value);
    void on_actionManual_Rotation_triggered();
    void on_SagView_clicked();
    void on_CorView_clicked();
    void on_AxiView_clicked();

    void on_remove_below_clicked();

    void on_show_bad_slice_clicked();
    void on_align_selected_dwi_clicked();
    void on_actionManual_Align_triggered();
    void on_actionT1W_based_QSDR_triggered();
    void on_change_fib_output_clicked();
    void on_fib_output_editingFinished();
};

#endif // RECONSTRUCTION_WINDOW_H

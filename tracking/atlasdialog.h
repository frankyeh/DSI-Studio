#ifndef ATLASDIALOG_H
#define ATLASDIALOG_H

#include <QDialog>

namespace Ui {
class AtlasDialog;
}
class tracking_window;
class AtlasDialog : public QDialog
{
    Q_OBJECT
    tracking_window& cur_tracking_window;
public:
    explicit AtlasDialog(tracking_window *parent);
    ~AtlasDialog();
    unsigned int index(void);
signals:
    void need_update(void);
private slots:
    void on_add_atlas_clicked();

    void on_atlasListBox_currentIndexChanged(int index);

    void on_pushButton_clicked();

private:
    Ui::AtlasDialog *ui;
};

#endif // ATLASDIALOG_H

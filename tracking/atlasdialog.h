#ifndef ATLASDIALOG_H
#define ATLASDIALOG_H
#include <memory>
#include <QDialog>

class fib_data;
namespace Ui {
class AtlasDialog;
}
class AtlasDialog : public QDialog
{
    Q_OBJECT
public:
    explicit AtlasDialog(QWidget *parent,std::shared_ptr<fib_data> handle_);
    ~AtlasDialog();
    std::shared_ptr<fib_data> handle;
    unsigned int atlas_index;
    std::string atlas_name;
    std::vector<unsigned int> roi_list;
    std::vector<std::string> roi_name;
signals:
    void need_update();
private slots:
    void on_add_atlas_clicked();

    void on_atlasListBox_currentIndexChanged(int index);

    void on_pushButton_clicked();

    void on_search_atlas_textChanged(const QString &arg1);

    void on_add_all_regions_clicked();

    void on_select_clicked();

    void on_merge_and_add_clicked();

private:
    Ui::AtlasDialog *ui;
};

#endif // ATLASDIALOG_H

#ifndef DB_OPERATION_DIALOG_H
#define DB_OPERATION_DIALOG_H

#include "connectometry/group_connectometry_analysis.h"
#include <QDialog>

namespace Ui {
class match_db;
}

class match_db : public QDialog
{
    Q_OBJECT

    std::shared_ptr<group_connectometry_analysis> vbc;
    void show_match_table(void);
public:
    explicit match_db(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc);
    ~match_db();

private slots:
    void on_buttonBox_accepted();

    void on_load_match_clicked();

    void on_match_consecutive_clicked();

private:
    Ui::match_db *ui;
};

#endif // DB_OPERATION_DIALOG_H

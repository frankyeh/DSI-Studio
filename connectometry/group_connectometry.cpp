#include <QMessageBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QStringListModel>
#include <QComboBox>
#include <functional>
#include <type_traits>
#include "tracking/region/Regions.h"
#include "group_connectometry.hpp"
#include "ui_group_connectometry.h"
#include "ui_tracking_window.h"
#include "tracking/tracking_window.h"
#include "tracking/tract/tracttablewidget.h"
#include "libs/tracking/fib_data.hpp"
#include "tracking/atlasdialog.h"
#include "tracking/roi.hpp"
#include "mainwindow.h"
QWidget *ROIViewDelegate::createEditor(QWidget *parent,
                                     const QStyleOptionViewItem &option,
                                     const QModelIndex &index) const
{
    if (index.column() == 2)
    {
        QComboBox *comboBox = new QComboBox(parent);
        comboBox->addItem("ROI");
        comboBox->addItem("ROA");
        comboBox->addItem("End");
        comboBox->addItem("Seed");
        comboBox->addItem("Terminative");
        connect(comboBox, SIGNAL(activated(int)), this, SLOT(emitCommitData()));
        return comboBox;
    }
    else
        return QItemDelegate::createEditor(parent,option,index);

}

void ROIViewDelegate::setEditorData(QWidget *editor,
                                  const QModelIndex &index) const
{

    if (index.column() == 2)
        ((QComboBox*)editor)->setCurrentIndex(index.model()->data(index).toString().toInt());
    else
        return QItemDelegate::setEditorData(editor,index);
}

void ROIViewDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                 const QModelIndex &index) const
{
    if (index.column() == 2)
        model->setData(index,QString::number(((QComboBox*)editor)->currentIndex()));
    else
        QItemDelegate::setModelData(editor,model,index);
}

void ROIViewDelegate::emitCommitData()
{
    emit commitData(qobject_cast<QWidget *>(sender()));
}


group_connectometry::group_connectometry(QWidget *parent,std::shared_ptr<group_connectometry_analysis> vbc_,QString db_file_name_) :
    QDialog(parent),
    null_pos_chart(new QChart),null_neg_chart(new QChart),
    null_pos_chart_view(new QChartView(null_pos_chart)),null_neg_chart_view(new QChartView(null_neg_chart)),
    fdr_chart(new QChart),fdr_chart_view(new QChartView(fdr_chart)),
    db_file_name(db_file_name_),work_dir(QFileInfo(db_file_name_).absoluteDir().absolutePath()),
    vbc(vbc_),db(vbc->handle->db),
    ui(new Ui::group_connectometry)
{

    ui->setupUi(this);
    // each button's objectName is exactly the command() name it should trigger
    for(auto* button : {ui->open_mr_files,ui->run,ui->show_result,ui->load_roi_from_atlas,
                         ui->clear_all_roi,ui->load_roi_from_file,ui->show_cohort,ui->apply_selection})
        connect(button,&QPushButton::clicked,this,&group_connectometry::on_button_command_clicked);
    ui->thread_count->setValue(tipl::max_thread_count);
    ui->chart_widget_layout->addWidget(null_pos_chart_view);
    ui->chart_widget_layout->addWidget(null_neg_chart_view);
    ui->chart_widget_layout->addWidget(fdr_chart_view);
    null_pos_chart->setMargins(QMargins(0,0,0,0));
    null_pos_chart->setBackgroundRoundness(0);
    null_neg_chart->setMargins(QMargins(0,0,0,0));
    null_neg_chart->setBackgroundRoundness(0);
    fdr_chart->setMargins(QMargins(0,0,0,0));
    fdr_chart->setBackgroundRoundness(0);

    setMouseTracking(true);
    ui->roi_table->setItemDelegate(new ROIViewDelegate(ui->roi_table));
    ui->roi_table->setAlternatingRowColors(true);


    ui->dist_table->setColumnCount(7);
    ui->dist_table->setColumnWidth(0,100);
    ui->dist_table->setColumnWidth(1,100);
    ui->dist_table->setColumnWidth(2,100);
    ui->dist_table->setColumnWidth(3,100);
    ui->dist_table->setColumnWidth(4,100);
    ui->dist_table->setColumnWidth(5,100);
    ui->dist_table->setColumnWidth(6,100);

    ui->length_threshold->setMaximum(vbc->handle->dim[0]);
    ui->length_threshold->setValue((vbc->handle->dim[0]/4)/5*5);


    // dist report
    ui->foi_widget->hide();
    on_roi_whole_brain_toggled(true);

    selected_count = db.subject_names.size();
    for(const auto& each : db.index_list)
        ui->index_name->addItem(QString::fromStdString(each));

    ui->subject_demo->clear();
    ui->subject_demo->setColumnCount(1);
    ui->subject_demo->setHorizontalHeaderLabels(QStringList("Subject ID"));
    ui->subject_demo->setRowCount(db.subject_names.size());
    for(unsigned int row = 0;row < ui->subject_demo->rowCount();++row)
        ui->subject_demo->setItem(row,0,new QTableWidgetItem(QString(db.subject_names[row].c_str())));

    ui->advanced_options->setCurrentIndex(0);

    if(!db.demo.empty())
        load_demographics();
    on_effect_size_valueChanged(0.3);
}

group_connectometry::~group_connectometry()
{
    delete ui;
}

template<typename T>
QLineSeries* get_line_series(const T& data, const char* name,QColor color,Qt::PenStyle s = Qt::SolidLine)
{
    QLineSeries* series = new QLineSeries;
    series->setName(name);
    QPen pen(color);
    pen.setWidth(2);
    pen.setStyle(s);
    series->setPen(pen);
    auto max_size = data.size()-1;
    while(max_size > 0 && (data[max_size] == 0.0f || data[max_size] == 1.0f))
        --max_size;
    ++max_size;
    for(size_t i = 0;i < max_size;++i)
        series->append(i,double(data[i]));
    return series;
}
void group_connectometry::show_fdr_report()
{
    if(vbc->fdr_inc.empty())
        return;
    fdr_chart->removeAllSeries();
    fdr_chart->addSeries(get_line_series(vbc->fdr_inc,vbc->hypothesis_inc.c_str(),0x00F01010));
    fdr_chart->addSeries(get_line_series(vbc->fdr_dec,vbc->hypothesis_dec.c_str(),0x001010F0));
    fdr_chart->createDefaultAxes();
    fdr_chart->axes(Qt::Horizontal).back()->setMin(vbc->length_threshold_voxels);
    fdr_chart->axes(Qt::Horizontal).back()->setTitleText("Length (voxel distance)");
    fdr_chart->axes(Qt::Vertical).back()->setTitleText("FDR");
    fdr_chart->axes(Qt::Horizontal).back()->setGridLineVisible(false);
    fdr_chart->axes(Qt::Vertical).back()->setGridLineVisible(false);
    fdr_chart->axes(Qt::Vertical).back()->setRange(0,1);

    fdr_chart->setTitle("FDR versus Track Length");
    ((QValueAxis*)fdr_chart->axes(Qt::Horizontal).back())->setTickType(QValueAxis::TicksDynamic);
    ((QValueAxis*)fdr_chart->axes(Qt::Horizontal).back())->setTickInterval(10);
}



void group_connectometry::show_report()
{
    if(vbc->tract_count_inc_null.empty())
        return;

    null_pos_chart->removeAllSeries();
    null_neg_chart->removeAllSeries();
    null_pos_chart->addSeries(get_line_series(vbc->tract_count_inc_null,"permuted",0x00F0A0A0,Qt::DashLine));
    null_pos_chart->addSeries(get_line_series(vbc->tract_count_inc,"nonpermuted",0x00F01010));
    null_neg_chart->addSeries(get_line_series(vbc->tract_count_dec_null,"permuted",0x00A0A0F0,Qt::DashLine));
    null_neg_chart->addSeries(get_line_series(vbc->tract_count_dec,"nonpermuted",0x001010F0));
    null_pos_chart->createDefaultAxes();
    null_pos_chart->axes(Qt::Horizontal).back()->setTitleText("Length (voxel distance)");
    null_pos_chart->axes(Qt::Horizontal).back()->setMin(vbc->length_threshold_voxels);
    null_pos_chart->axes(Qt::Vertical).back()->setTitleText("Count");
    null_pos_chart->axes(Qt::Horizontal).back()->setGridLineVisible(false);
    null_pos_chart->axes(Qt::Vertical).back()->setGridLineVisible(false);
    null_pos_chart->setTitle(vbc->hypothesis_inc.c_str());
    null_neg_chart->createDefaultAxes();
    null_neg_chart->axes(Qt::Horizontal).back()->setTitleText("Length (voxel distance)");
    null_neg_chart->axes(Qt::Horizontal).back()->setMin(vbc->length_threshold_voxels);
    null_neg_chart->axes(Qt::Vertical).back()->setTitleText("Count");
    null_neg_chart->axes(Qt::Horizontal).back()->setGridLineVisible(false);
    null_neg_chart->axes(Qt::Vertical).back()->setGridLineVisible(false);
    null_neg_chart->setTitle(vbc->hypothesis_dec.c_str());
    ((QValueAxis*)null_pos_chart->axes(Qt::Horizontal).back())->setTickType(QValueAxis::TicksDynamic);
    ((QValueAxis*)null_pos_chart->axes(Qt::Horizontal).back())->setTickInterval(10);
    ((QValueAxis*)null_neg_chart->axes(Qt::Horizontal).back())->setTickType(QValueAxis::TicksDynamic);
    ((QValueAxis*)null_neg_chart->axes(Qt::Horizontal).back())->setTickInterval(10);
}

void group_connectometry::show_dis_table(void)
{
    ui->dist_table->setRowCount(100);
    ui->dist_table->setHorizontalHeaderLabels(
                QStringList() << "voxel spacing"
                << QString("FDR (%1)").arg(vbc->hypothesis_inc.c_str())
                << QString("FDR (%1)").arg(vbc->hypothesis_dec.c_str())
                << QString("#Tracts(%1)(permuted)").arg(vbc->hypothesis_inc.c_str())
                << QString("#Tracts(%1)(permuted)").arg(vbc->hypothesis_dec.c_str())
                << QString("#Tracts(%1)(nonpermuted)").arg(vbc->hypothesis_inc.c_str())
                << QString("#Tracts(%1)(nonpermuted)").arg(vbc->hypothesis_dec.c_str()));
    for(unsigned int index = vbc->length_threshold_voxels;index < vbc->fdr_inc.size()-1;++index)
    {
        int row = int(index-vbc->length_threshold_voxels);
        ui->dist_table->setItem(row,0,new QTableWidgetItem(QString::number(index)));
        ui->dist_table->setItem(row,1, new QTableWidgetItem(QString::number(double(vbc->fdr_inc[index]))));
        ui->dist_table->setItem(row,2, new QTableWidgetItem(QString::number(double(vbc->fdr_dec[index]))));
        ui->dist_table->setItem(row,3, new QTableWidgetItem(QString::number(vbc->tract_count_inc_null[index])));
        ui->dist_table->setItem(row,4, new QTableWidgetItem(QString::number(vbc->tract_count_dec_null[index])));
        ui->dist_table->setItem(row,5, new QTableWidgetItem(QString::number(vbc->tract_count_inc[index])));
        ui->dist_table->setItem(row,6, new QTableWidgetItem(QString::number(vbc->tract_count_dec[index])));
    }
    ui->dist_table->selectRow(0);
}

void group_connectometry::on_button_command_clicked()
{
    if(auto* button = qobject_cast<QPushButton*>(sender()))
        command({button->objectName().toStdString()},command_source::User);
}


void fill_demo_table(const connectometry_db& db,
                     QTableWidget* table)
{
    QStringList t2;
    t2 << "Subject";
    for(size_t i = 0;i < db.titles.size();++i)
        t2 << db.titles[i].c_str();
    table->clear();
    table->setColumnCount(t2.size());
    table->setHorizontalHeaderLabels(t2);
    table->setRowCount(int(db.subject_names.size()));
    for(size_t row = 0;row < db.subject_names.size();++row)
    {
        table->setItem(int(row),0,new QTableWidgetItem(QString(db.subject_names[row].c_str())));
        for(size_t col = 0;col < db.titles.size();++col)
        {
            auto item_pos = size_t(row)*db.titles.size()+col;
            if(item_pos < db.items.size())
                table->setItem(int(row),int(col)+1,new QTableWidgetItem(QString(db.items[item_pos].c_str())));
            else
                table->setItem(int(row),int(col)+1,new QTableWidgetItem(QString()));
        }
    }
}

void group_connectometry::load_demographics(void)
{
    model.reset(new stat_model);
    model->read_demo(db);
    // fill up regression values
    {
        QStringList t;
        for(size_t i = 0; i < db.feature.size();++i)
            t << QString::fromStdString(db.feature[i].title);
        ui->variable_list->clear();
        ui->variable_list->addItems(t);
        for(int i = 0;i < ui->variable_list->count();++i)
        {
            ui->variable_list->item(i)->setFlags(ui->variable_list->item(i)->flags() | Qt::ItemIsUserCheckable); // set checkable flag
            ui->variable_list->item(i)->setCheckState(i == 0 ? Qt::Checked : Qt::Unchecked);
            ui->variable_list->setItemAlignment(Qt::AlignLeft);
        }
        ui->cohort_index->clear();
        ui->cohort_index->addItems(t);
        ui->cohort_index->setCurrentIndex(0);
        ui->foi_widget->show();
    }
    on_variable_list_clicked(QModelIndex());
    fill_demo_table(db,ui->subject_demo);
}

struct group_connectometry_param
{
    std::string id;
    std::function<QString(void)> get;
    std::function<bool(QString)> set; // returns false on malformed/out-of-range input; the widget is left unchanged
    QComboBox* combo = nullptr; // set for dropdown-backed params, so list_param can show their options
};

static std::vector<group_connectometry_param> get_settable_params(Ui::group_connectometry* ui)
{
    // one generic setter for every widget kind below; if constexpr picks the right parse/apply
    // logic for whichever concrete type "box" is deduced as at each call site
    auto set_value = [](auto* box)
    {
        return [box](QString v)->bool
        {
            using T = std::remove_pointer_t<decltype(box)>;
            bool ok = true;
            if constexpr(std::is_same_v<T,QLineEdit>)
                box->setText(v);
            else if constexpr(std::is_same_v<T,QComboBox>)
            {
                int i = v.toInt(&ok);
                if(ok && (i < 0 || i >= box->count()))
                    ok = false;
                if(ok)
                    box->setCurrentIndex(i);
            }
            else if constexpr(std::is_base_of_v<QAbstractButton,T>) // QCheckBox and QRadioButton (e.g. roi_whole_brain)
            {
                int i = v.toInt(&ok);
                if(ok)
                    box->setChecked(i != 0);
            }
            else if constexpr(std::is_same_v<T,QDoubleSpinBox>)
            {
                double d = v.toDouble(&ok);
                if(ok)
                    box->setValue(d);
            }
            else // QSpinBox
            {
                int i = v.toInt(&ok);
                if(ok)
                    box->setValue(i);
            }
            return ok;
        };
    };
    return {
        {"no_tractogram",[ui]{return QString::number(ui->no_tractogram->isChecked());},set_value(ui->no_tractogram)},
        {"index_name",[ui]{return QString::number(ui->index_name->currentIndex());},set_value(ui->index_name),ui->index_name},
        {"foi",[ui]{return QString::number(ui->foi->currentIndex());},set_value(ui->foi),ui->foi},
        {"length_threshold",[ui]{return QString::number(ui->length_threshold->value());},set_value(ui->length_threshold)},
        {"tip",[ui]{return QString::number(ui->tip->value());},set_value(ui->tip)},
        {"fdr_control",[ui]{return QString::number(ui->fdr_control->isChecked());},set_value(ui->fdr_control)},
        {"fdr_threshold",[ui]{return QString::number(ui->fdr_threshold->value());},set_value(ui->fdr_threshold)},
        {"threshold",[ui]{return QString::number(ui->threshold->value());},set_value(ui->threshold)},
        {"effect_size",[ui]{return QString::number(ui->effect_size->value());},set_value(ui->effect_size)},
        {"region_pruning",[ui]{return QString::number(ui->region_pruning->isChecked());},set_value(ui->region_pruning)},
        {"normalize_iso",[ui]{return QString::number(ui->normalize_iso->isChecked());},set_value(ui->normalize_iso)},
        {"output_name",[ui]{return ui->output_name->text();},set_value(ui->output_name)},
        {"exclude_cerebellum",[ui]{return QString::number(ui->exclude_cb->isChecked());},set_value(ui->exclude_cb)},
        {"roi_whole_brain",[ui]{return QString::number(ui->roi_whole_brain->isChecked());},set_value(ui->roi_whole_brain)},
        {"thread_count",[ui]{return QString::number(ui->thread_count->value());},set_value(ui->thread_count)},
        {"permutation_count",[ui]{return QString::number(ui->permutation_count->value());},set_value(ui->permutation_count)},
        {"select_text",[ui]{return ui->select_text->text();},set_value(ui->select_text)},
    };
}

bool group_connectometry::command(std::vector<std::string> cmd,command_source source)
{
    if(cmd.empty())
        return tipl::error() << (error_msg = "empty command"),false;
    cmd.resize(3);
    std::string name = cmd[0],param = cmd[1];
    error_msg.clear();
    auto fail = [&](const std::string& msg)->bool
    {
        error_msg = msg;
        tipl::error() << error_msg;
        if(source == command_source::User)
            QMessageBox::critical(this,"ERROR",error_msg.c_str());
        return false;
    };

    if(name == "list_param")
    {
        auto params = get_settable_params(ui);
        auto id = QString::fromStdString(param).trimmed().toLower();
        id.replace('-','_');
        if(id.isEmpty() || id == "all")
        {
            tipl::out() << "id\tvalue";
            for(auto& p : params)
                tipl::out() << p.id << "\t" << p.get().toStdString();
            return true;
        }
        for(auto& p : params)
            if(id == p.id.c_str())
            {
                tipl::out() << p.id << "\t" << p.get().toStdString();
                if(p.combo)
                {
                    QStringList options;
                    for(int i = 0;i < p.combo->count();++i)
                        options << QString("%1=%2").arg(i).arg(p.combo->itemText(i));
                    tipl::out() << "options: " << options.join(", ").toStdString();
                }
                return true;
            }
        return fail("invalid parameter: "+param+"; use list_param without an argument to list all parameters");
    }

    if(name == "set_param" || name == "set_params")
    {
        auto params = get_settable_params(ui);
        auto set_one = [&](const std::string& id,const std::string& value)->bool
        {
            for(auto& p : params)
                if(p.id == id)
                    return p.set(QString::fromStdString(value));
            return false;
        };
        if(name == "set_param")
        {
            if(!set_one(param,cmd[2]))
                return fail("invalid parameter: "+param);
        }
        else
            for(const auto& kv : tipl::split(param,'&'))
            {
                auto pos = kv.find('=');
                if(pos == std::string::npos)
                    return fail("invalid parameter: "+kv);
                if(!set_one(kv.substr(0,pos),kv.substr(pos+1)))
                    return fail("invalid parameter: "+kv.substr(0,pos));
            }
        return true;
    }

    if(name == "open_mr_files")
    {
        if(param.empty())
        {
            if(source != command_source::User)
                return fail("please specify a demographics file");
            QString filename = QFileDialog::getOpenFileName(
                        this,"Open demographics",work_dir,
                        "Comma- or Tab-Separated Values(*.csv *.tsv);;Text File(*.txt);;All files (*)");
            if(filename.isEmpty())
                return false; // canceled, not an error
            param = filename.toStdString();
        }
        if(!db.parse_demo(param))
            return fail(db.handle->error_msg);
        load_demographics();
        return true;
    }

    if(name == "run")
    {
        if(ui->run->text() == "Stop")
        {
            vbc->clear();
            timer->stop();
            timer.reset();
            ui->progressBar->setValue(0);
            ui->run->setText("Run");
            return true;
        }
        // longitudinal data without loading demographics
        if(db.is_longitudinal && !model.get())
        {
            model.reset(new stat_model);
            model->read_demo(vbc->handle->db);
        }
        if(!model.get())
            return fail("Load demographic file first");

        // check cohort text
        if(!command({"show_cohort"},source))
            return false;
        if(model->remove_list.empty()) // select cohort failed
            return fail("cohort selection failed");

        // setup parameters
        {
            vbc->no_tractogram = ui->no_tractogram->isChecked();
            vbc->foi_str = ui->foi->currentText().toStdString();
            vbc->handle->db.set_current_index(ui->index_name->currentIndex());
            vbc->length_threshold_voxels = uint32_t(ui->length_threshold->value());
            vbc->tip_iteration = uint32_t(ui->tip->value());
            vbc->fdr_threshold = ui->fdr_control->isChecked() ? float(ui->fdr_threshold->value()) : 0.0f;
            vbc->t_threshold = float(ui->threshold->value());
            vbc->rho_threshold = float(ui->effect_size->value());
            vbc->region_pruning = ui->region_pruning->isChecked();
            vbc->normalize_iso = db.is_longitudinal ? false : ui->normalize_iso->isChecked();
            vbc->output_file_name = ui->output_name->text().toStdString();
        }

        // setup statistical model
        {
            vbc->model.reset(new stat_model);
            *(vbc->model.get()) = *(model.get());
            if(!vbc->model->select_feature(db,ui->foi->currentText().toStdString()))
                return fail(vbc->model->error_msg);
        }

        // setup roi
        {
            vbc->roi_mgr = std::make_shared<RoiMgr>(vbc->handle);
            if(ui->exclude_cb->isChecked())
                vbc->exclude_cerebellum();

            // apply ROI
            if(!ui->roi_whole_brain->isChecked())
            {
                std::vector<unsigned char> roi_type(roi_list.size());
                std::vector<std::string> roi_name(roi_list.size());
                for(unsigned int index = 0;index < roi_list.size();++index)
                {
                    roi_type[index] = uint8_t(ui->roi_table->item(int(index),2)->text().toInt());
                    roi_name[index] = ui->roi_table->item(int(index),0)->text().toStdString();
                }
                for(unsigned int index = 0;index < roi_list.size();++index)
                    vbc->roi_mgr->setRegions(roi_list[index],roi_type[index],roi_name[index].c_str());
            }

            // if no seed assigned, assign whole brain
            if(vbc->roi_mgr->seeds.empty())
                vbc->roi_mgr->setWholeBrainSeed(vbc->fiber_threshold);
        }

        vbc->run_permutation(uint32_t(ui->thread_count->value()),uint32_t(ui->permutation_count->value()));

        ui->run->setText("Stop");
        timer.reset(new QTimer(this));
        timer->setInterval(1000);
        connect(timer.get(), SIGNAL(timeout()), this, SLOT(calculate_FDR()));
        timer->start();
        return true;
    }

    if(name == "show_result")
    {
        if(!vbc->model.get())
            return fail("run the analysis first");
        std::shared_ptr<fib_data> new_data(new fib_data);
        *(new_data.get()) = *(vbc->handle);
        {
            result_fib.reset(new connectometry_result);
            stat_model info;
            info.resample(*(vbc->model.get()),false,false,0);
            vbc->calculate_spm(*result_fib.get(),info);
            new_data->slices.push_back(std::make_shared<slice_model>("dec_t",result_fib->dec_ptr[0],new_data->dim));
            new_data->slices.push_back(std::make_shared<slice_model>("inc_t",result_fib->inc_ptr[0],new_data->dim));
        }
        tracking_window* current_tracking_window = new tracking_window(this,new_data);
        if(auto* mw = qobject_cast<MainWindow*>(parentWidget()))
            mw->report_and_target_window(current_tracking_window);
        current_tracking_window->set_memorize_parameters(false);
        current_tracking_window->setAttribute(Qt::WA_DeleteOnClose);
        current_tracking_window->setWindowTitle(vbc->output_file_name.c_str());
        current_tracking_window->showNormal();
        current_tracking_window->tractWidget->addNewTracts(vbc->hypothesis_inc.c_str());
        current_tracking_window->tractWidget->addNewTracts(vbc->hypothesis_dec.c_str());

        current_tracking_window->tractWidget->tract_models[0]->add(*(vbc->inc_track.get()));
        current_tracking_window->tractWidget->tract_models[1]->add(*(vbc->dec_track.get()));

        for(const auto& each : std::vector<std::pair<std::string, std::string>>{
            {"show_surface", "1"},{"show_slice", "0"},{"show_region", "0"},{"bkg_color", "16777215"},{"surface_alpha", "0.2"}})
                current_tracking_window->set_data(each.first.c_str(),each.second.c_str());

        current_tracking_window->command({"set_zoom","0.8"});
        current_tracking_window->command({"add_surface","0","25"});
        current_tracking_window->command({"update_tract"});
        return true;
    }

    if(name == "load_roi_from_atlas")
    {
        if(source != command_source::User)
            return fail("load_roi_from_atlas requires interactive selection");
        if(vbc->handle->atlas_list.empty())
            return fail("no atlas available");
        std::shared_ptr<AtlasDialog> atlas_dialog(new AtlasDialog(this,vbc->handle));
        if(atlas_dialog->exec() != QDialog::Accepted)
            return false; // canceled, not an error
        for(unsigned int i = 0;i < atlas_dialog->roi_list.size();++i)
        {
            std::vector<tipl::vector<3,short> > points;
            if(!vbc->handle->get_atlas_roi(atlas_dialog->atlas_name,atlas_dialog->roi_name[i],points))
                return fail("cannot get atlas ROI: "+atlas_dialog->roi_name[i]);
            add_new_roi(atlas_dialog->roi_name[i].c_str(),atlas_dialog->atlas_name.c_str(),points);
        }
        return true;
    }

    if(name == "clear_all_roi")
    {
        roi_list.clear();
        ui->roi_table->setRowCount(0);
        return true;
    }

    if(name == "load_roi_from_file")
    {
        if(param.empty())
        {
            if(source != command_source::User)
                return fail("please specify a NIFTI ROI file");
            QString file = tipl::qt::open_image_file(this,work_dir + "/roi.nii.gz","Report file (*.nii *nii.gz);;Text files (*.txt);;All files (*)");
            if(file.isEmpty())
                return false; // canceled, not an error
            param = file.toStdString();
        }
        tipl::image<3> I;
        tipl::matrix<4,4> transform;
        std::string nifti_error;
        if(!(tipl::io::gz_nifti(param,std::ios::in)
                >> transform >> I
                >> [&nifti_error](const std::string& e){nifti_error = e;}))
            return fail(nifti_error);
        transform.inv();
        transform *= vbc->handle->trans_to_mni;
        std::vector<tipl::vector<3,short> > new_roi;
        for (tipl::pixel_index<3> index(vbc->handle->dim);index < vbc->handle->dim.size();++index)
        {
            tipl::vector<3> pos(index);
            pos.to(transform);
            pos.round();
            if(!I.shape().is_valid(pos) || I.at(pos) == 0)
                continue;
            new_roi.push_back(tipl::vector<3,short>((const unsigned int*)index.begin()));
        }
        if(new_roi.empty())
            return fail("The nifti contain no voxel with value greater than 0.");
        add_new_roi(QFileInfo(param.c_str()).baseName(),"Local File",new_roi);
        return true;
    }

    if(name == "show_cohort")
    {
        if(!model.get())
            return fail("load demographic file first");
        if(!model->select_cohort(db,ui->select_text->text().toStdString()))
            return fail(model->error_msg);
        selected_count = 0;
        ui->subject_demo->setUpdatesEnabled(false);
        for(size_t i = 0;i < model->remove_list.size();++i)
        {
            if(!model->remove_list[i])
                selected_count++;
            for(int j = 0;j < ui->subject_demo->columnCount();++j)
                ui->subject_demo->item(int(i),j)->setBackground(model->remove_list[i] ? Qt::white : QColor(255,255,200));
        }
        ui->subject_demo->setUpdatesEnabled(true);
        ui->cohort_report->setText(QString("n=%1").arg(selected_count));

        ui->run->setEnabled(selected_count > 2);
        ui->effect_size->setEnabled(selected_count > 2);
        ui->threshold->setEnabled(selected_count > 2);
        tipl::out() << "n=" << selected_count;
        return true;
    }

    if(name == "list_voi")
    {
        tipl::out() << "index\tname\tselected";
        for(size_t i = 0;i < db.feature.size();++i)
            tipl::out() << i << "\t" << db.feature[i].title << "\t" << (db.feature[i].selected ? "1" : "0");
        return true;
    }

    if(name == "set_voi")
    {
        if(cmd[2].empty())
            return fail("usage: set_voi <voi> <variable_list>; names or indices from list_voi, comma-separated");
        auto foi_str = db.select_voi(param,cmd[2]);
        if(foi_str.empty())
            return fail(db.handle->error_msg);
        sync_variable_list();
        ui->foi->setCurrentText(QString::fromStdString(foi_str));
        tipl::out() << "variable of interest: " << foi_str;
        return true;
    }

    if(name == "get_demo")
    {
        tipl::out() << "subject\t" << tipl::merge(db.titles,'\t');
        for(size_t row = 0;row < db.subject_names.size();++row)
        {
            std::ostringstream out;
            out << db.subject_names[row];
            for(size_t col = 0;col < db.titles.size();++col)
            {
                auto pos = row*db.titles.size()+col;
                out << "\t" << (pos < db.items.size() ? db.items[pos] : std::string());
            }
            tipl::out() << out.str();
        }
        return true;
    }

    if(name == "apply_selection")
    {
        QString new_text(ui->select_text->text());
        if(!new_text.isEmpty())
            new_text += ",";
        if(param.empty())
        {
            new_text += ui->cohort_index->currentText();
            new_text += (ui->cohort_operator->currentIndex() == 3 ? QString("/") : ui->cohort_operator->currentText());
            new_text += ui->cohort_value->text();
        }
        else
            new_text += param.c_str();
        ui->select_text->setText(new_text);
        return command({"show_cohort"},source);
    }

    return fail("unknown command: "+name);
}

void group_connectometry::calculate_FDR(void)
{
    if(vbc->prog == 100 && timer.get())
        timer->stop();

    ui->progressBar->setValue(vbc->prog);
    vbc->calculate_FDR();
    show_report();
    show_dis_table();
    show_fdr_report();


    int pos = ui->textBrowser->verticalScrollBar()->value();
    ui->textBrowser->setHtml(vbc->generate_report().c_str());
    ui->textBrowser->verticalScrollBar()->setValue(pos);

    if(vbc->prog < 100)
        return;

    // tipl::progress = 100
    {
        tipl::progress prog("output distribution image");
        delete null_pos_chart_view;
        delete null_neg_chart_view;
        delete fdr_chart_view;
        null_pos_chart = new QChart;
        null_pos_chart_view = new QChartView(null_pos_chart);
        null_neg_chart = new QChart;
        null_neg_chart_view = new QChartView(null_neg_chart);
        fdr_chart = new QChart;
        fdr_chart_view = new QChartView(fdr_chart);
        null_pos_chart->setMargins(QMargins(0,0,0,0));
        null_pos_chart->setBackgroundRoundness(0);
        null_neg_chart->setMargins(QMargins(0,0,0,0));
        null_neg_chart->setBackgroundRoundness(0);
        fdr_chart->setMargins(QMargins(0,0,0,0));
        fdr_chart->setBackgroundRoundness(0);
        show_report();
        show_fdr_report();
        null_pos_chart_view->grab().save((vbc->output_file_name+".inc.dist.jpg").c_str());
        null_neg_chart_view->grab().save((vbc->output_file_name+".dec.dist.jpg").c_str());
        fdr_chart_view->grab().save((vbc->output_file_name+".fdr.jpg").c_str());
        ui->chart_widget_layout->addWidget(null_pos_chart_view);
        ui->chart_widget_layout->addWidget(null_neg_chart_view);
        ui->chart_widget_layout->addWidget(fdr_chart_view);


        if(vbc->inc_track->get_visible_track_count() ||
           vbc->dec_track->get_visible_track_count())
            QMessageBox::information(this,QApplication::applicationName(),"tractography saved");
        else
            QMessageBox::information(this,QApplication::applicationName(),"no significant finding");

        ui->run->setText("Run");
        ui->progressBar->setValue(100);
        timer.reset();
    }
}

void group_connectometry::on_roi_whole_brain_toggled(bool checked)
{
    ui->roi_table->setEnabled(!checked);
    ui->load_roi_from_atlas->setEnabled(!checked);
    ui->clear_all_roi->setEnabled(!checked);
    ui->load_roi_from_file->setEnabled(!checked);
}

void group_connectometry::add_new_roi(QString name,QString source,
                                      const std::vector<tipl::vector<3,short> >& new_roi,
                                      int type)
{
    ui->roi_table->setRowCount(ui->roi_table->rowCount()+1);
    ui->roi_table->setItem(ui->roi_table->rowCount()-1,0,new QTableWidgetItem(name));
    ui->roi_table->setItem(ui->roi_table->rowCount()-1,1,new QTableWidgetItem(source));
    ui->roi_table->setItem(ui->roi_table->rowCount()-1,2,new QTableWidgetItem(QString::number(type)));
    ui->roi_table->openPersistentEditor(ui->roi_table->item(ui->roi_table->rowCount()-1,2));
    roi_list.push_back(new_roi);
}

void group_connectometry::on_variable_list_clicked(const QModelIndex &)
{
    for(int i = 0;i < ui->variable_list->count();++i)
        db.feature[uint32_t(i)].selected = (ui->variable_list->item(i)->checkState() == Qt::Checked);
    sync_variable_list();
}
void group_connectometry::sync_variable_list(void)
{
    for(int i = 0;i < ui->variable_list->count();++i)
        ui->variable_list->item(i)->setCheckState(db.feature[uint32_t(i)].selected ? Qt::Checked : Qt::Unchecked);
    auto foi_str = ui->foi->currentText();
    ui->foi->clear();
    for(int i = 0;i < ui->variable_list->count();++i)
        if(db.feature[uint32_t(i)].selected)
            ui->foi->addItem(ui->variable_list->item(i)->text());
    if(db.is_longitudinal && db.longitudinal_filter_type == 0)
        ui->foi->addItem(QString("longitudinal change"));
    ui->foi->setCurrentText(foi_str);
}

void group_connectometry::on_fdr_control_toggled(bool checked)
{
    ui->fdr_threshold->setEnabled(checked);
}

void group_connectometry::on_effect_size_valueChanged(double rho)
{
    ui->threshold->blockSignals(true);
    ui->threshold->setValue(rho*std::sqrt(double(selected_count)-2)/(1-rho*rho));
    ui->threshold->blockSignals(false);
}


void group_connectometry::on_threshold_valueChanged(double t)
{
    ui->effect_size->blockSignals(true);
    ui->effect_size->setValue(t/std::sqrt(t*t+selected_count-2));
    ui->effect_size->blockSignals(false);
}

bool can_be_normalized_by_iso(const std::string& name);
void group_connectometry::on_index_name_currentIndexChanged(int index)
{
    if(can_be_normalized_by_iso(ui->index_name->currentText().toStdString()) && !db.is_longitudinal)
        ui->normalize_iso->show();
    else
        ui->normalize_iso->hide();
}


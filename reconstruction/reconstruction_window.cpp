#include <filesystem>
#include <QSplitter>
#include <QSignalBlocker>
#include <QThread>
#include <QImage>
#include <QMessageBox>
#include <QInputDialog>
#include <QFileDialog>
#include <QSettings>
#include <QPainter>
#include <QButtonGroup>
#include <QStyle>
#include <QStyleOption>
#include "reconstruction_window.h"
#include "ui_reconstruction_window.h"

#include "mainwindow.h"
#include "tracking/tracking_window.h"
#include "tracking/region/Regions.h"
#include "libs/dsi/image_model.hpp"
#include "manual_alignment.h"

void populate_templates(QComboBox* combo,size_t index);
void move_current_dir_to(const std::string& file_name);

std::string to_cmd(const std::string& cmd);
bool eddy_check_shell(const std::vector<float>& bvalues,std::string& cause);

bool reconstruction_window::load_src(int index)
{
    handle = std::make_shared<src_data>();
    if (!handle->load_from_file(tipl::qt::to_path(filenames[index])))
        return false;
    move_current_dir_to(filenames[index].toStdString());
    tipl::progress prog("initiate interface");
    existing_steps = handle->voxel.steps;
    if(handle->voxel.is_histology)
        return true;
    update_dimension();
    double m = double(tipl::max_value(handle->src_dwi_data[0]));
    ui->max_value->setMaximum(m*1.5);
    ui->max_value->setMinimum(0.0);
    ui->max_value->setSingleStep(m*0.05);
    ui->max_value->setValue(m*0.2f);
    ui->min_value->setMaximum(m*1.5);
    ui->min_value->setMinimum(0.0);
    ui->min_value->setSingleStep(m*0.05);
    ui->min_value->setValue(0.0);
    load_b_table();

    return true;
}

QString check_citation(QString str);
extern std::vector<std::filesystem::path> qa_template_list,iso_template_list;
reconstruction_window::reconstruction_window(QStringList filenames_,QWidget *parent) :
    QMainWindow(parent),filenames(filenames_),ui(new Ui::reconstruction_window)
{
    ui->setupUi(this);
    for(int i = 0;i < 3;++i)
    {
        QPixmap pix(20,20);
        pix.fill(Qt::transparent);
        QPainter painter(&pix);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(i == 0 ? "#15803d" : i == 1 ? "#059669" : "#16a34a"));
        painter.drawEllipse(1,1,18,18);
        painter.setPen(Qt::white);
        QFont font = painter.font();
        font.setBold(true);
        font.setPointSize(8);
        painter.setFont(font);
        painter.drawText(pix.rect(),Qt::AlignCenter,QString("2%1").arg(QChar('a'+i)));
        ui->nav_rail->item(i)->setIcon(QIcon(pix));
    }
    // Navigation includes a disclosure row without a corresponding page.
    ui->nav_rail->item(4)->setHidden(true);
    ui->nav_rail->item(5)->setHidden(true);
    ui->nav_rail->item(3)->setForeground(QColor("#2563eb"));
    connect(ui->nav_rail,&QListWidget::currentRowChanged,this,[this](int row)
    {
        if(row >= 0 && row < 3)
            ui->pages->setCurrentIndex(row);
        if(row >= 4 && row <= 5)
            ui->pages->setCurrentIndex(row-1);
    });
    connect(ui->nav_rail,&QListWidget::itemClicked,this,[this](QListWidgetItem* item)
    {
        if(item != ui->nav_rail->item(3))
            return;
        ui->nav_rail->item(4)->setHidden(false);
        ui->nav_rail->item(5)->setHidden(handle->voxel.is_histology);
        ui->nav_rail->item(3)->setHidden(true);
        ui->nav_rail->setCurrentRow(4);
    });
    connect(ui->volume_list,&QListWidget::currentRowChanged,this,[this](int row)
    {
        if(row >= 0 && row != ui->b_table->currentRow())
            ui->b_table->selectRow(row);
    });
    connect(ui->b_table,&QTableWidget::itemSelectionChanged,this,[this]()
    {
        if(ui->volume_list->currentRow() != ui->b_table->currentRow())
            ui->volume_list->setCurrentRow(ui->b_table->currentRow());
    });
    connect(ui->b_table->model(),&QAbstractItemModel::dataChanged,this,[this](const QModelIndex& first,const QModelIndex& last)
    {
        for(int row = first.row();row <= last.row() && row < ui->volume_list->count();++row)
            if(row >= 0 && ui->b_table->item(row,0) && ui->b_table->item(row,1) &&
               ui->b_table->item(row,2) && ui->b_table->item(row,3))
                ui->volume_list->item(row)->setText(QString("#%1  b%2  (%3, %4, %5)")
                    .arg(row+1).arg(ui->b_table->item(row,0)->text())
                    .arg(ui->b_table->item(row,1)->text().toDouble(),0,'f',3)
                    .arg(ui->b_table->item(row,2)->text().toDouble(),0,'f',3)
                    .arg(ui->b_table->item(row,3)->text().toDouble(),0,'f',3));
    });
    outputs = {
        ui->output_fa,
        ui->output_rd,
        ui->output_md,
        ui->output_ad,
        ui->output_rd12,
        ui->output_tensor,
        ui->output_helix,
        ui->output_rdi,
        ui->output_gfa,
        ui->output_odf,
        ui->high_accuracy // fixel
    };
    adv_outputs = {
        ui->output_rd12,
        ui->output_tensor,
        ui->output_helix,
        ui->output_gfa,
        ui->output_odf,
        ui->dti_ignore_high_b
    };
    for(auto each : adv_outputs)
        each->hide();
    if(!load_src(0))
        throw std::runtime_error(handle->error_msg);

    setWindowTitle(filenames[0]);
    ui->ThreadCount->setMaximum(tipl::max_thread_count);
    ui->graphicsView->setScene(&scene);
    ui->view_source->setScene(&source);
    ui->b_table->setColumnWidth(0,60);
    ui->b_table->setColumnWidth(1,80);
    ui->b_table->setColumnWidth(2,80);
    ui->b_table->setColumnWidth(3,80);
    ui->b_table->setHorizontalHeaderLabels(QStringList() << "b value" << "bx" << "by" << "bz");

    populate_templates(ui->primary_template,handle->voxel.template_id);
    ui->diffusion_sampling->setValue(handle->voxel.param[0]);
    v2c.two_color(tipl::rgb(0,0,0),tipl::rgb(255,255,255));

    absolute_path = QFileInfo(filenames[0]).absolutePath();

    ui->fib_output_widget->setVisible(filenames.size() == 1);
    switch(settings.value("rec_method_id",int(handle->voxel.method_id)).toInt())
    {
    case 1:
        ui->DTI->setChecked(true);
        on_DTI_toggled(true);
        break;
    case 7:
        ui->QSDR->setChecked(true);
        on_QSDR_toggled(true);
        break;
    default:
        ui->GQI->setChecked(true);
        on_GQI_toggled(true);
        break;
    }

    ui->ThreadCount->setValue(handle->voxel.thread_count);

    ui->report->setText(check_citation(handle->voxel.report.c_str()));
    ui->dti_ignore_high_b->setChecked(handle->voxel.dti_ignore_high_b);

    ui->method_group->setVisible(!handle->voxel.is_histology);
    ui->hist_param_group->setVisible(handle->voxel.is_histology);

    ui->qsdr_reso->setValue(handle->voxel.qsdr_reso);

    if(handle->voxel.is_histology)
    {
        for(auto* each : {ui->actionSave_SRC_file_as,ui->actionSave_4D_nifti,ui->actionSave_b0,ui->actionSave_DWI_sum,
                          ui->actionSave_b_table,ui->actionSave_bvals,ui->actionSave_bvecs})
            each->setVisible(false);
        for(int row : {1,5}) // Preprocessing and Edit Btable
            ui->nav_rail->item(row)->setHidden(true);
        // Inspect / Edit DWI keeps only the image flips that apply to histology
        for(auto* each : std::initializer_list<QWidget*>{ui->viewer_card,ui->signal_group,
                          ui->btn_Flip_xz,ui->btn_Flip_yz,ui->btn_Align_ACPC,
                          ui->btn_Manual_Rotation,ui->btn_Resample,ui->btn_Rotate,ui->btn_Overwrite_Voxel_Size,ui->btn_Attach_Images})
            each->hide();

        ui->hist_downsampling->setValue(std::ceil(std::log2(handle->voxel.hist_image.width()))-12);
    }


    pills = {{ui->q1_yes,ui->q1_no},{ui->q2_yes,ui->q2_no},{ui->q3_yes,ui->q3_no},
             {ui->q4_yes,ui->q4_no},{ui->q5_yes,ui->q5_no},{ui->q6_yes,ui->q6_no},{ui->q7_yes,ui->q7_no}};
    preproc = {{ui->btn_Run_FSL_Topup,"topup_eddy","applytopup",0,6,1},
               {ui->btn_TOPUP_shell,"topup","applytopup",0,6,2},
               {ui->btn_TOPUP_only,"topup","applytopup",1,-1,0},
               {ui->btn_Eddy_Motion_Correction,"motion_correction","Motion correction",1,2,1},
               {ui->btn_EDDY,"eddy","using FSL eddy",2,6,1},
               {ui->btn_T1w_Distortion_Correction,"correct_by_t1w","inverted b0",2,3,1},
               {ui->btn_Motion_DSI,"motion_correction","Motion correction",3,2,1},
               {ui->btn_T1w_DSI,"correct_by_t1w","inverted b0",3,3,1},
               {ui->btn_Correct_Bias_Field,"bias_field_correction","bias field",-1,4,2},
               {ui->btn_Resample,"resample","resampled to",-1,5,1}};
    for(const auto& each : pills)
    {
        auto* group = new QButtonGroup(this); // always one answer
        group->addButton(each.first);
        group->addButton(each.second);
        connect(group,&QButtonGroup::buttonClicked,this,[this,no = each.second](QAbstractButton* b)
        {
            if(b == ui->q1_yes || b == ui->q4_yes) // the reversed phase-encoding b0 or the T1w
            {
                auto file = tipl::qt::open_image_file(this,absolute_path,b == ui->q1_yes ?
                            "Reversed phase-encoding images (*.sz *.rz *src.gz *.nii *nii.gz);;All files (*)" :
                            "T1w images (*nii.gz);;All files (*)");
                if(file.isEmpty()) // no file selected: back to No
                    no->setChecked(true);
                else
                    (b == ui->q1_yes ? rev_pe_file : t1w_file) = file.toStdString();
            }
            update_preproc();
        });
    }
    connect(ui->btn_Run_Preprocessing,&QPushButton::clicked,this,[this]()
    {
        for(const auto& s : preproc)
        {
            std::string cmd(s.cmd);
            if(s.button->property("act").toBool() &&
               !command({cmd,tipl::contains(cmd,"topup") ? rev_pe_file : cmd == "correct_by_t1w" ? t1w_file : std::string()},command_source::User))
                return;
        }
    });
    if(!handle->voxel.is_histology) // default answers from the b-table and voxel size
    {
        const auto& vs = handle->voxel.vs;
        (std::abs(vs[0]-vs[1]) > 0.01f || std::abs(vs[1]-vs[2]) > 0.01f ? ui->q6_yes : ui->q6_no)->setChecked(true);
        std::string cause;
        std::vector<float> shells;
        for(auto b : handle->src_bvalues)
            if(b > 50.0f && std::none_of(shells.begin(),shells.end(),[b](float s){return std::abs(s-b) < 100.0f;}))
                shells.push_back(b);
        bool shelled = eddy_check_shell(handle->src_bvalues,cause);
        (shelled ? ui->q2_yes : ui->q2_no)->setChecked(true);
        ui->preproc_chip->setText("Your data: " + (!shelled ? QString("DSI (%1 b-values)").arg(shells.size()) :
                                  shells.size() == 1 ? QString("single-shell") : QString("multi-shell (%1 shells)").arg(shells.size())));
    }
    ui->preproc_flow->installEventFilter(this);
    update_preproc();

    connect(ui->z_pos,SIGNAL(valueChanged(int)),this,SLOT(on_b_table_itemSelectionChanged()));
    connect(ui->max_value,SIGNAL(valueChanged(double)),this,SLOT(on_b_table_itemSelectionChanged()));
    connect(ui->min_value,SIGNAL(valueChanged(double)),this,SLOT(on_b_table_itemSelectionChanged()));

    on_b_table_itemSelectionChanged();

    ui->image_group->hide();
    ui->signal_group->hide();
    connect(ui->image_group_toggle,&QPushButton::clicked,this,[this]()
    {
        ui->image_group_toggle->hide();
        ui->image_group->show();
    });
    connect(ui->signal_group_toggle,&QPushButton::clicked,this,[this]()
    {
        ui->signal_group_toggle->hide();
        ui->signal_group->show();
    });
    ui->method_detail->hide();
    connect(ui->method_more_toggle,&QPushButton::toggled,this,[this](bool checked)
    {
        ui->method_detail->setVisible(checked);
        if(checked)
            for(auto each : adv_outputs)
                each->show();
        ui->method_more_toggle->setText(checked ? "▾ Less..." : "▸ More...");
    });

    {
        foreach (QAction* action, findChildren<QAction*>())
            if(action->toolTip().startsWith("run "))
                connect(action,&QAction::triggered,this,[this,action](){command({action->toolTip().mid(4).section('=',0,0).toStdString(),action->toolTip().mid(4).section('=',1).toStdString()},command_source::User);});
        foreach (QPushButton* pb, findChildren<QPushButton*>())
            if(pb->toolTip().startsWith("run "))
                connect(pb,&QPushButton::clicked,this,[this,pb](){command({pb->toolTip().mid(4).section('=',0,0).toStdString(),pb->toolTip().mid(4).section('=',1).toStdString()},command_source::User);});
    }
}
void reconstruction_window::update_dimension(void)
{
    source_ratio = std::max(1.0,500/(double)handle->voxel.dim.height());
    if(ui->SlicePos->maximum() != handle->voxel.dim[2]-1)
    {
        ui->SlicePos->setRange(0,handle->voxel.dim[2]-1);
        ui->SlicePos->setValue((handle->voxel.dim[2]-1) >> 1);
    }
    if(ui->z_pos->maximum() != handle->voxel.dim[view_orientation]-1)
    {
        ui->z_pos->setRange(0,handle->voxel.dim[view_orientation]-1);
        ui->z_pos->setValue((handle->voxel.dim[view_orientation]-1) >> 1);
    }
}

void reconstruction_window::update_preproc(void)
{
    auto done = [this](const char* text){return tipl::contains(handle->voxel.report,text);}; // the report records the applied steps
    int a[7]; // 1: yes, 2: no
    for(size_t i = 0;i < 7;++i)
        a[i] = pills[i].first->isChecked() ? 1 : 2;
    path = (a[0]-1)*2+a[1]-1;
    const auto& vs = handle->voxel.vs;
    bool distortion_done = done("applytopup") || done("inverted b0");
    bool enabled[7] = {!distortion_done,true,
                       !done("Motion correction") && !done("using FSL eddy") && path % 2,
                       !distortion_done && path >= 2,
                       !done("bias field"),
                       !done("resampled to") && (std::abs(vs[0]-vs[1]) > 0.01f || std::abs(vs[1]-vs[2]) > 0.01f), // not yet isotropic
                       a[1] == 1 && !done("using FSL eddy")}; // FSL eddy needs shells
    for(size_t i = 0;i < 7;++i)
        pills[i].first->parentWidget()->setEnabled(enabled[i]);
    ui->btn_Run_FSL_Topup->setVisible(a[6] == 1); // topup/eddy or topup alone, by the eddy question
    ui->btn_TOPUP_shell->setVisible(a[6] == 2);
    ui->q6_note->setText(QString("%1×%2×%3 mm").arg(vs[0]).arg(vs[1]).arg(vs[2]));

    auto set = [](QWidget* w,const char* name,bool value)
    {
        if(w->property(name).toBool() == value)
            return;
        w->setProperty(name,value);
        w->style()->unpolish(w);
        w->style()->polish(w);
    };
    for(const auto& s : preproc) // a step runs on its path when its question, if any, is enabled and answered as required
    {
        bool d = done(s.report);
        s.button->setText((d ? "✓ " : "") + s.button->text().remove("✓ "));
        set(s.button,"done",d);
        set(s.button,"act",!d && (s.path < 0 || s.path == path) && (s.q < 0 || (enabled[s.q] && a[s.q] == s.when)));
        set(s.button,"fade",s.path >= 0 && s.path != path);
    }
    ui->br_yes->setEnabled(path < 2);
    ui->br_no->setEnabled(path >= 2);
    QLabel* br[4] = {ui->br_s1,ui->br_s2,ui->br_s3,ui->br_s4};
    for(int i = 0;i < 4;++i)
        br[i]->setEnabled(path == i);
    ui->preproc_flow->update();
}
bool reconstruction_window::eventFilter(QObject* obj,QEvent* event)
{
    if(obj != ui->preproc_flow || event->type() != QEvent::Paint)
        return QMainWindow::eventFilter(obj,event);
    QPainter p(ui->preproc_flow);
    QStyleOption opt;
    opt.initFrom(ui->preproc_flow);
    ui->preproc_flow->style()->drawPrimitive(QStyle::PE_Widget,&opt,&p,ui->preproc_flow);
    p.setRenderHint(QPainter::Antialiasing);
    auto font = p.font();
    font.setPointSize(8);
    font.setBold(true);
    p.setFont(font);
    auto line = [&](double x1,double y1,double x2,double y2,bool on)
    {
        p.setPen(QPen(QColor(on ? "#059669" : "#c8d6ce"),on ? 2.0 : 1.5));
        p.setBrush(p.pen().color());
        p.drawLine(QPointF(x1,y1),QPointF(x2,y2));
    };
    auto arrow = [&](double x,double y1,double y2,bool on,const char* text)
    {
        line(x,y1,x,y2-1,on);
        p.drawPolygon(QPolygonF({QPointF(x-4,y2-7),QPointF(x,y2-1),QPointF(x+4,y2-7)}));
        if(text)
            p.drawText(QPointF(x+6,(y1+y2)/2+4),text);
    };
    auto g = [](QWidget* w){return QRectF(w->geometry());};
    QRectF br[2] = {g(ui->br_yes),g(ui->br_no)},s[4] = {g(ui->br_s1),g(ui->br_s2),g(ui->br_s3),g(ui->br_s4)};
    // Q1 fork, then Q2 forks under each answer
    double y = br[0].top()-8,x0 = br[0].center().x(),x1 = br[1].center().x(),mid = (x0+x1)/2;
    line(x0,y,mid,y,path < 2);
    line(mid,y,x1,y,path >= 2);
    line(x0,y,x0,br[0].top(),path < 2);
    line(x1,y,x1,br[1].top(),path >= 2);
    for(int side = 0;side < 2;++side)
    {
        double x = br[side].center().x(),ty = s[side*2].top()-6,xa = s[side*2].center().x(),xb = s[side*2+1].center().x();
        line(x,br[side].bottom()+2,x,ty,path/2 == side);
        line(xa,ty,x,ty,path == side*2);
        line(x,ty,xb,ty,path == side*2+1);
        line(xa,ty,xa,s[side*2].top(),path == side*2);
        line(xb,ty,xb,s[side*2+1].top(),path == side*2+1);
    }
    // each column down to a shared bus, then the shared steps down to Run Preprocessing
    QRectF bias = g(ui->btn_Correct_Bias_Field),resample = g(ui->btn_Resample),run = g(ui->btn_Run_Preprocessing);
    double bus = bias.top()-16,cx = bias.center().x();
    line(s[0].center().x(),bus,s[3].center().x(),bus,false);
    for(int i = 0;i < 4;++i)
    {
        double x = s[i].center().x(),yy = s[i].bottom()+3;
        for(const auto& st : preproc)
            if(st.path == i && st.button->isVisible())
            {
                arrow(x,yy,g(st.button).top(),path == i,st.q < 0 || st.q == 6 ? nullptr : "Yes");
                yy = g(st.button).bottom();
            }
        line(x,yy,x,bus,path == i);
    }
    line(s[path].center().x(),bus,cx,bus,true);
    arrow(cx,bus,bias.top(),true,"No");
    arrow(cx,bias.bottom(),resample.top(),true,"Yes");
    arrow(cx,resample.bottom(),run.top(),true,nullptr);
    return true;
}
void reconstruction_window::load_b_table(void)
{
    if(handle->src_bvalues.empty())
        return;
    QSignalBlocker block(ui->volume_list);
    ui->volume_list->clear();
    ui->b_table->clear();
    ui->b_table->setRowCount(handle->src_bvalues.size());
    for(unsigned int index = 0;index < handle->src_bvalues.size();++index)
    {
        ui->b_table->setItem(index,0, new QTableWidgetItem(QString::number(handle->src_bvalues[index])));
        ui->b_table->setItem(index,1, new QTableWidgetItem(QString::number(handle->src_bvectors[index][0],'f',3)));
        ui->b_table->setItem(index,2, new QTableWidgetItem(QString::number(handle->src_bvectors[index][1],'f',3)));
        ui->b_table->setItem(index,3, new QTableWidgetItem(QString::number(handle->src_bvectors[index][2],'f',3)));
        ui->volume_list->addItem(QString("#%1  b%2  (%3, %4, %5)")
            .arg(index+1).arg(handle->src_bvalues[index])
            .arg(handle->src_bvectors[index][0],0,'f',3)
            .arg(handle->src_bvectors[index][1],0,'f',3)
            .arg(handle->src_bvectors[index][2],0,'f',3));
    }
    ui->b_table->selectRow(0);
    ui->volume_list->setCurrentRow(0);
}

void reconstruction_window::on_b_table_itemSelectionChanged()
{
    if(size_t(ui->b_table->currentRow()) >= handle->src_dwi_data.size())
        return;
    v2c.set_range(ui->min_value->value(),ui->max_value->value());

    QImage source_image;
    source_image << v2c[
                    tipl::volume2slice_scaled(
                       handle->src_dwi_data[ui->b_table->currentRow()],
                       view_orientation,ui->z_pos->value(),
                       source_ratio)];

    if(view_orientation != 2)
    {
        // show bad_slices
        if(bad_slice_analyzed)
        {
            std::vector<size_t> mark_slices;
            for(size_t index = 0;index < bad_slices.size();++index)
                if(bad_slices[index].first == ui->b_table->currentRow())
                    mark_slices.push_back(bad_slices[index].second);
            for(size_t index = 0;index < mark_slices.size();++index)
            {
                QPainter paint(&source_image);
                paint.setPen(Qt::red);
                paint.drawLine(0,source_ratio*mark_slices[index],
                               source_image.width(),source_ratio*mark_slices[index]);
            }
        }
        source_image = source_image.mirrored();
    }
    source << source_image;
}


void reconstruction_window::resizeEvent ( QResizeEvent * event )
{
    QMainWindow::resizeEvent(event);
    ui->horizontalLayout_5->setDirection(width() < 900 ? QBoxLayout::TopToBottom : QBoxLayout::LeftToRight);
    ui->viewer_split->setDirection(width() < 800 ? QBoxLayout::TopToBottom : QBoxLayout::LeftToRight);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}
void reconstruction_window::showEvent ( QShowEvent * event )
{
    QMainWindow::showEvent(event);
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::closeEvent(QCloseEvent *event)
{
    QMainWindow::closeEvent(event);

}

reconstruction_window::~reconstruction_window()
{
    delete ui;
}


void reconstruction_window::on_actionSave_Mask_triggered()
{
    QString filename = tipl::qt::save_image_file(this,absolute_path+"/mask.nii.gz",
            "Nifti file(*nii.gz *.nii);;Text files (*.txt);;All files (*)" );
    if(filename.isEmpty())
        return;
    if(!filename.toLower().endsWith(".txt"))
        filename = QFileInfo(filename).absolutePath() + "/" + QFileInfo(filename).baseName() + ".nii.gz";
    ROIRegion region(handle->dwi.shape(),handle->voxel.vs);
    region.from_mask(handle->voxel.mask);
    region.save_region_to_file(filename.toStdString());
}

bool reconstruction_window::command(std::vector<std::string> cmds,command_source source)
{
    if(cmds.empty())
        return tipl::error() << (error_msg = "empty command"),false;
    if(cmds.size() > 2)
        return tipl::error() << (error_msg = "too many parameters"),false;
    cmds.resize(2);
    std::string cmd = to_cmd(cmds[0]),param = cmds[1];

    error_msg.clear();
    auto fail = [&](const std::string& msg = std::string())
    {
        if(!msg.empty())
            handle->error_msg = msg;
        error_msg = handle->error_msg;
        tipl::error() << error_msg;
        if(source == command_source::User && !error_msg.empty() && !tipl::ends_with(error_msg,{"canceled","aborted"})) // a user cancel needs no dialog
            QMessageBox::critical(this,"ERROR",error_msg.c_str());
        return false;
    };
    if(cmd == "reconstruction")
        return fail("use recon");
    if(cmd == "recon")
    {
        if(source == command_source::User)
        {
            auto confirm = [&](bool condition,const QString& question,std::vector<std::string> correction_cmd)
            {
                if(!condition)
                    return true;
                auto result = QMessageBox::information(this,QApplication::applicationName(),question,
                                QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel);
                if(result == QMessageBox::Yes)
                    handle->command(correction_cmd);
                return result != QMessageBox::Cancel;
            };
            if(!confirm(handle->voxel.vs[2] > handle->voxel.vs[0]*1.2f && handle->is_human_data && !ui->QSDR->isChecked(), // non isotropic resolution
                "The slice thickness is much larger than slice resolution. This is not ideal for fiber tracking. Resample slice thickness to 2mm isotropic resolution?",
                {"resample","2"}) ||
               !confirm(!tipl::contains(handle->voxel.report,"bias field"),
                "Correct signal inhomogeneity using bias field correction?",
                {"bias_field_correction"}) ||
               !confirm(!handle->is_human_data && (handle->long_axis_direction() != 1),
                "This seems to be an animal scan in non-axial orientation. Correct image orientation?",
                {"orientation_correction"}))
                return fail("canceled");
        }

        unsigned char method_id = source == command_source::User ?
            (ui->DTI->isChecked() ? 1 : ui->QSDR->isChecked() ? 7 : 4) : handle->voxel.method_id;
        if(!param.empty())
            method_id = uint8_t(std::stoi(param));
        if(method_id != handle->voxel.method_id)
        {
            handle->voxel.method_id = method_id;
            handle->output_file_name = handle->file_name;
            handle->check_output_file_name();
            if(filenames.size() == 1)
                ui->fib_output->setText(tipl::qt::to_qstring(handle->output_file_name));
        }

        if(method_id == 7 && qa_template_list.empty())
            return fail("cannot find template files");

        std::string ref_file_name = handle->file_name.u8string();
        std::string ref_output_file_name = handle->output_file_name.u8string();
        std::string ref_steps(handle->voxel.steps.begin()+existing_steps.length(),handle->voxel.steps.end());
        std::shared_ptr<src_data> ref_handle = handle;
        std::string ref_existing_steps = existing_steps;
        std::vector<std::filesystem::path> output_files;
        bool ok = true;
        auto report_file_error = [&](int index,const std::string& msg)
        {
            ok = false;
            error_msg = QFileInfo(filenames[index]).fileName().toStdString() + " : " + msg;
            tipl::error() << error_msg;
            if(source == command_source::User && !tipl::ends_with(msg,{"canceled","aborted"}))
                QMessageBox::critical(this,"ERROR",error_msg.c_str());
        };
        tipl::progress prog("process SRC files",true);
        for(int index = 0;prog(index,filenames.size());++index)
        {
            tipl::out() << "processing " << filenames[index].toStdString() << std::endl;
            if(index)
            {
                if(!load_src(index) || !handle->run_steps(ref_file_name,ref_steps))
                {
                    ok = false;
                    if(!prog.aborted())
                        report_file_error(index,handle->error_msg);
                    break;
                }
                if(!ref_output_file_name.empty())
                {
                    std::string derived_output;
                    if(!tipl::match_files(ref_file_name,handle->file_name.u8string(),ref_output_file_name,derived_output))
                    {
                        report_file_error(index,"cannot derive output file name from "+ref_output_file_name);
                        break;
                    }
                    handle->output_file_name = derived_output;
                }
            }

            if(tipl::max_value(handle->voxel.mask) == 0)
            {
                report_file_error(index,"please select mask for reconstruction");
                break;
            }

            handle->voxel.method_id = method_id;
            if(source == command_source::User)
            {
                settings.setValue("rec_method_id",method_id);
                handle->voxel.dti_ignore_high_b = ui->dti_ignore_high_b->isChecked();
                handle->voxel.thread_count = ui->ThreadCount->value();
                handle->voxel.template_id = ui->primary_template->currentIndex();
                if(method_id == 4 || method_id == 7)
                {
                    handle->voxel.param[0] = float(ui->diffusion_sampling->value());
                    settings.setValue("rec_gqi_sampling",ui->diffusion_sampling->value());
                }

                handle->voxel.other_output.clear();
                for(auto each : outputs)
                    if(each->isChecked())
                    {
                        if(!handle->voxel.other_output.empty())
                            handle->voxel.other_output += ",";
                        handle->voxel.other_output += each->statusTip().toStdString();
                    }

                if(handle->voxel.is_histology)
                {
                    handle->voxel.vs[0] = handle->voxel.vs[1] = handle->voxel.vs[2] = float(ui->hist_resolution->value());
                    handle->voxel.hist_downsampling = uint32_t(ui->hist_downsampling->value());
                    handle->voxel.hist_raw_smoothing = uint32_t(ui->hist_raw_smoothing->value());
                    handle->voxel.hist_tensor_smoothing = uint32_t(ui->hist_tensor_smoothing->value());
                }

                if(method_id == 7)
                    handle->voxel.qsdr_reso = ui->qsdr_reso->value();
            }

            auto dim_backup = handle->voxel.dim;
            auto vs = handle->voxel.vs;
            auto trans = handle->voxel.trans_to_mni;
            auto mask = handle->voxel.mask;
            ok = handle->reconstruction();
            handle->voxel.mask = mask;
            handle->voxel.dim = dim_backup;
            handle->voxel.vs = vs;
            handle->voxel.trans_to_mni = trans;

            if(!ok)
            {
                report_file_error(index,handle->error_msg);
                break;
            }
            output_files.push_back(handle->output_file_name);
        }
        handle = ref_handle;
        existing_steps = ref_existing_steps;
        update_dimension();
        load_b_table();
        on_SlicePos_valueChanged(ui->SlicePos->value());

        if(!ok)
            return false;

        for(const auto& output : output_files)
        {
            tipl::out() << "reconstruction output: " << output;
            ((MainWindow*)parent())->addFib(tipl::qt::to_qstring(output));
        }

        if(source == command_source::User)
        {
            QMessageBox::information(this,QApplication::applicationName(),"FIB file created");
            raise();
        }
        return true;
    }
    if(cmd == "regenerate_btable" && param.empty() && source == command_source::User) // uses the selected template's reference
        handle->voxel.template_id = ui->primary_template->currentIndex();
    if(cmd == "mask_from_template" || cmd == "mask_from_unet") // both pick the model by the selected template
    {
        if(param.empty())
            handle->voxel.template_id = source == command_source::User ? ui->primary_template->currentIndex() : handle->voxel.template_id;
        else
        {
            size_t v = size_t(std::stoll(param));
            if(v >= qa_template_list.size() || v >= iso_template_list.size())
                return fail("invalid template index");
            handle->voxel.template_id = v;
        }
    }
    if((cmd == "resample" || cmd == "align_acpc") && param.empty())
    {
        bool ok;
        float nv = float(QInputDialog::getDouble(this,
            QApplication::applicationName(),"Assign output resolution in (mm):", double(handle->voxel.vs[0]),0.0,3.0,4, &ok));
        if (!ok || nv == 0.0f)
            return fail("canceled");
        param = std::to_string(nv);
    }
    if(tipl::begins_with(cmd,"save_") && param.empty())
    {
        static const std::map<std::string,std::pair<QString,QString> > save_types{
            {"save_src",{".sz","SRC files (*.sz *src.gz);;All files (*)"}},
            {"save_b_table",{".b_table.txt","Text files (*.txt);;All files (*)"}},
            {"save_bval",{".bval","All files (*)"}},
            {"save_bvec",{".bvec","All files (*)"}}};
        auto type = save_types.count(cmd) ? save_types.at(cmd) : std::pair<QString,QString>{".nii.gz","NIFTI files (*nii.gz);;All files (*)"};
        QString filename = tipl::qt::save_image_file(this,QFileInfo(filenames[0]).baseName()+type.first,type.second);
        if(filename.isEmpty())
            return fail("canceled");
        param = filename.toStdString();
    }
    if(cmd == "save_src" && source == command_source::User) // the displayed mask is reviewed by the user
        handle->apply_mask = true;
    if((cmd == "partial_fov" || cmd == "set_voxel_size") && param.empty())
    {
        bool ok;
        const auto& vs = handle->voxel.vs;
        param = QInputDialog::getText(this,QApplication::applicationName(),
                    cmd == "partial_fov" ? "MNI coordinate range (minx miny minz maxx maxy maxz)" : "Voxel size in mm",QLineEdit::Normal,
                    cmd == "partial_fov" ? QString("-36 -30 -20 36 30 24") : QString("%1 %2 %3").arg(vs[0]).arg(vs[1]).arg(vs[2]),&ok).toStdString();
        if(!ok)
            return fail("canceled");
    }
    if((cmd == "mask_background_signals" || cmd == "probabilistic_masking" || cmd == "correct_by_t1w" || cmd == "correct_by_t2w") && param.empty())
    {
        QString filename = tipl::qt::open_image_file(this,QFileInfo(filenames[0]).baseName() + ".nii.gz",
                                "NIFTI files (*nii.gz);;All files (*)" );
        if(filename.isEmpty())
            return fail("canceled");
        param = filename.toStdString();
    }
    if(cmd == "correct_by_t1w" && source == command_source::User && param.find('|') == std::string::npos)
    {
        bool ok;
        QStringList items{"Native","1.5 mm isotropic","1.0 mm isotropic"};
        auto item = QInputDialog::getItem(this,QApplication::applicationName(),"Output resolution:",items,1,false,&ok);
        if(!ok)
            return fail("canceled");
        param += std::vector<std::string>{"|0","|1.5","|1.0"}[size_t(items.indexOf(item))];
    }
    if(source == command_source::User &&
       tipl::contains_case_insensitive(cmd,"topup") && !std::filesystem::exists(handle->corrected_file()) && param.empty())
    {
        auto candidate = tipl::remove_all_suffix(handle->file_name);
        candidate += ".rz";
        if(!std::filesystem::exists(candidate))
        {
            QMessageBox::information(this,QApplication::applicationName(),"Please specify another nii.gz or sz file with reversed phase encoding data");
            auto other_src = tipl::qt::open_image_file(
                    this,absolute_path,"Images (*.sz *.rz *src.gz *.nii *nii.gz);;DICOM image (*.dcm);;All files (*)" );
            if(other_src.isEmpty())
                return fail("canceled");
            param = other_src.toStdString();
        }
        else
            param = candidate.u8string();
    }

    if(((tipl::contains_case_insensitive(cmd,"open") || tipl::begins_with(cmd,"load_bv") ||
        cmd == "add_other_image") && param.empty()) || (cmd == "regenerate_btable" && param == "src")) // "src": pick the reference SRC
    {
        QString filename = tipl::qt::open_image_file(this,absolute_path,
            cmd == "add_other_image" ? "Images (*.nii *nii.gz);;All files (*)" :
            cmd == "regenerate_btable" ? "SRC files (*.sz *src.gz);;All files (*)" :
            tipl::begins_with(cmd,"load_bv") ? "All files (*)" : "Mask files (*.nii *nii.gz *.hdr);;Text files (*.txt);;All files (*)");
        if(filename.isEmpty())
            return fail("canceled");
        param = filename.toStdString();
    }

    bool result = handle->command({cmd,param});
    if(!result)
        fail();
    else
        ui->report->setText(check_citation(handle->voxel.report.c_str()));
    update_preproc();

    update_dimension();
    load_b_table();
    on_SlicePos_valueChanged(ui->SlicePos->value());


    if(result &&
       filenames.size() > 1 &&
       tipl::begins_with(cmd,"save_") &&
        QMessageBox::information(this,QApplication::applicationName(),"Apply to other SRC files?",
        QMessageBox::Yes|QMessageBox::No|QMessageBox::Cancel) == QMessageBox::Yes)
    {
        tipl::progress prog("apply to other SRC files");
        std::string previous_steps(handle->voxel.steps.begin()+existing_steps.length(),handle->voxel.steps.end());
        for(int index = 1;prog(index,filenames.size());++index)
        {
            src_data model;
            if (!model.load_from_file(filenames[index].toStdString()) ||
                !model.run_steps(handle->file_name.u8string(),previous_steps + cmd + "=" + param + "\n"))
            {
                if(QMessageBox::critical(this,QApplication::applicationName(),
                    QFileInfo(filenames[index]).fileName() + " : " + model.error_msg.c_str() + " Continue?",
                                QMessageBox::Yes|QMessageBox::No) == QMessageBox::No)
                    return tipl::error() << (error_msg = handle->error_msg = model.error_msg),false;
            }
        }
    }
    return result;
}
void reconstruction_window::on_DTI_toggled(bool checked)
{
    if(!checked)
        return;
    ui->DTI_metrics->setVisible(true);
    ui->high_accuracy->setVisible(!checked); // GQI and QSDR only
    if(checked)
        ui->high_accuracy->setChecked(false);
    ui->qsdr_options->setVisible(!checked);
    ui->GQIOption_2->setVisible(!checked);
    ui->ODF_metrics->setVisible(!checked);


    ui->qsdr_reso_row->setVisible(!checked);

    if(filenames.size() == 1)
    {
        handle->voxel.method_id = 1;
        handle->output_file_name = handle->file_name;
        handle->check_output_file_name();
        ui->fib_output->setText(tipl::qt::to_qstring(handle->output_file_name));
    }
}


void reconstruction_window::on_GQI_toggled(bool checked)
{
    if(!checked)
        return;
    ui->DTI_metrics->setVisible(false);
    ui->high_accuracy->setVisible(true);
    ui->qsdr_options->setVisible(!checked);
    ui->GQIOption_2->setVisible(checked);
    ui->ODF_metrics->setVisible(checked);


    ui->qsdr_reso_row->setVisible(!checked);

    if(filenames.size() == 1)
    {
        handle->voxel.method_id = 4;
        handle->output_file_name = handle->file_name;
        handle->check_output_file_name();
        ui->fib_output->setText(tipl::qt::to_qstring(handle->output_file_name));
    }

}

void reconstruction_window::on_QSDR_toggled(bool checked)
{
    if(!checked)
        return;
    ui->DTI_metrics->setVisible(false);
    ui->high_accuracy->setVisible(true);
    ui->qsdr_options->setVisible(checked);
    ui->GQIOption_2->setVisible(checked);
    ui->ODF_metrics->setVisible(checked);


    ui->qsdr_reso_row->setVisible(checked);

    if(filenames.size() == 1)
    {
        handle->voxel.method_id = 7;
        handle->output_file_name = handle->file_name;
        handle->check_output_file_name();
        ui->fib_output->setText(tipl::qt::to_qstring(handle->output_file_name));
    }
}

void reconstruction_window::on_zoom_in_clicked()
{
    source_ratio *= 1.1f;
    on_b_table_itemSelectionChanged();
}

void reconstruction_window::on_zoom_out_clicked()
{
    source_ratio *= 0.9f;
    on_b_table_itemSelectionChanged();
}


bool load_image_from_files(QStringList filenames,tipl::image<3>& ref,tipl::vector<3>& vs,tipl::matrix<4,4>&);
void reconstruction_window::on_actionRotate_triggered()
{
    QStringList filenames = tipl::qt::open_image_files(this,absolute_path,"Images (*.nii *nii.gz *.dcm);;All files (*)" );
    if( filenames.isEmpty())
        return;

    tipl::image<3> ref;
    tipl::vector<3> vs;
    tipl::matrix<4,4> t;
    if(!load_image_from_files(filenames,ref,vs,t))
        return;
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                                                tipl::reg::subject_image_pre(tipl::image<3>(handle->dwi)),tipl::image<3,unsigned char>(),handle->voxel.vs,
                                                                tipl::reg::template_image_pre(tipl::image<3>(ref)),tipl::image<3,unsigned char>(),vs,
                                                                tipl::reg::rigid_body,
                                                                tipl::reg::cost_type::mutual_info));
    manual->from_T = handle->voxel.trans_to_mni;
    manual->to_T = t;
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;

    tipl::progress prog_("rotating");
    tipl::image<3> ref2(ref);
    float m = tipl::median(ref2.begin(),ref2.end());
    ref *= 0.5f/m;
    handle->rotate(ref.shape(),vs,manual->get_iT());
    handle->voxel.report += " The diffusion images were rotated and scaled to the space of "
                         + QFileInfo(filenames[0]).baseName().toStdString()
                         + ". The b-table was also rotated accordingly.";
    ui->report->setText(handle->voxel.report.c_str());
    load_b_table();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());

}


void reconstruction_window::on_delete_2_clicked()
{
    if(ui->b_table->currentRow() >= 0 && command({"remove",std::to_string(ui->b_table->currentRow())},command_source::User))
        bad_slice_analyzed = false;
}

void reconstruction_window::on_remove_below_clicked()
{
    if(ui->b_table->currentRow() > 0 && command({"remove",std::to_string(ui->b_table->currentRow())+":end"},command_source::User))
        bad_slice_analyzed = false;
}


void reconstruction_window::on_SlicePos_valueChanged(int position)
{
    tipl::color_image buffer;
    handle->draw_mask(buffer,position);
    double ratio =
        std::min(double(ui->graphicsView->width()-5)/double(buffer.width()),
                 double(ui->graphicsView->height()-5)/double(buffer.height()));
    QImage view;
    view << buffer;
    view = view.scaled(int(buffer.width()*ratio),int(buffer.height()*ratio));
    scene << view;
}

void reconstruction_window::on_actionT1W_based_QSDR_triggered()
{
    QString subject_image = tipl::qt::open_image_file(
            this,absolute_path,
            "Images (*.nii *nii.gz);;All files (*)" );
    if(subject_image.isEmpty())
        return;
    QString template_image = tipl::qt::open_image_file(
            this,absolute_path,
            "Images (*.nii *nii.gz);;All files (*)" );
    if(template_image.isEmpty())
        return;
    if(handle->add_other_image("reg",subject_image.toStdString()))
    {
        handle->voxel.other_modality_template = template_image.toStdString();
        QMessageBox::information(this,QApplication::applicationName(),"Registration images added");
    }
    else
        QMessageBox::critical(this,"ERROR","Not a valid nifti file");

}

void reconstruction_window::on_actionManual_Rotation_triggered()
{
    std::shared_ptr<manual_alignment> manual(
                new manual_alignment(this,tipl::reg::subject_image_pre(tipl::image<3>(handle->dwi)),tipl::image<3,unsigned char>(),handle->voxel.vs,
                                          tipl::reg::subject_image_pre(tipl::image<3>(handle->dwi)),tipl::image<3,unsigned char>(),handle->voxel.vs,tipl::reg::rigid_body,tipl::reg::cost_type::mutual_info));
    manual->from_T = handle->voxel.trans_to_mni;
    manual->to_T = handle->voxel.trans_to_mni;
    if(manual->exec() != QDialog::Accepted)
        return;
    tipl::progress prog_("rotating");
    handle->rotate(handle->dwi.shape(),handle->voxel.vs,manual->get_iT());
    handle->voxel.report += " The diffusion images were manually rotated.";
    load_b_table();
    update_dimension();
    on_SlicePos_valueChanged(ui->SlicePos->value());
}

void reconstruction_window::on_SagView_clicked()
{
    view_orientation = 0;
    ui->z_pos->setRange(0,handle->voxel.dim[view_orientation]-1);
    ui->z_pos->setValue((handle->voxel.dim[view_orientation]-1) >> 1);
    on_b_table_itemSelectionChanged();
}

void reconstruction_window::on_CorView_clicked()
{
    view_orientation = 1;
    ui->z_pos->setRange(0,handle->voxel.dim[view_orientation]-1);
    ui->z_pos->setValue((handle->voxel.dim[view_orientation]-1) >> 1);
    on_b_table_itemSelectionChanged();
}

void reconstruction_window::on_AxiView_clicked()
{
    view_orientation = 2;
    ui->z_pos->setRange(0,handle->voxel.dim[view_orientation]-1);
    ui->z_pos->setValue((handle->voxel.dim[view_orientation]-1) >> 1);
    on_b_table_itemSelectionChanged();
}

void reconstruction_window::on_show_bad_slice_clicked()
{
    if(!bad_slice_analyzed)
    {
        bad_slices = handle->get_bad_slices();
        bad_slice_analyzed = true;
        std::vector<char> is_bad(ui->b_table->rowCount());
        for(int i = 0;i < bad_slices.size();++i)
            if(bad_slices[i].first < is_bad.size())
                is_bad[bad_slices[i].first] = 1;

        for(int i = 0;i < ui->b_table->rowCount();++i)
        {
            QColor color = is_bad[i] ? QColor(255,200,200) : QColor(255,255,255);
            for(int j = 0;j < ui->b_table->columnCount();++j)
                ui->b_table->item(i,j)->setData(Qt::BackgroundRole,color);
            if(i < ui->volume_list->count())
                ui->volume_list->item(i)->setBackground(color);
        }
    }
    if(bad_slices.size() == 0)
    {
        QMessageBox::information(this,QApplication::applicationName(),"No bad slice found in this data");
        return;
    }
    on_b_table_itemSelectionChanged();
    ui->bad_slice_label->setText(QString("A total %1 bad slices marked by red").arg(bad_slices.size()));

}

void reconstruction_window::on_align_selected_dwi_clicked()
{
    if(ui->b_table->currentRow() < 0)
        return;
    tipl::image<3> from(handle->src_dwi_data[ui->b_table->currentRow()]),to(handle->src_dwi_data[0]);
    tipl::normalize(from);
    tipl::normalize(to);
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                                                tipl::reg::subject_image_pre(std::move(from)),tipl::image<3,unsigned char>(),handle->voxel.vs,
                                                                tipl::reg::subject_image_pre(std::move(to)),tipl::image<3,unsigned char>(),handle->voxel.vs,
                                                                tipl::reg::rigid_body,
                                                                tipl::reg::cost_type::mutual_info));
    manual->from_T = handle->voxel.trans_to_mni;
    manual->to_T = handle->voxel.trans_to_mni;
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;

    handle->rotate_one_dwi(ui->b_table->currentRow(),manual->get_iT());

    update_dimension();
    load_b_table();
    on_SlicePos_valueChanged(ui->SlicePos->value());

}




void match_template_resolution(tipl::image<3>& VG,
                               tipl::vector<3>& VGvs,
                               tipl::image<3>& VF,
                               tipl::vector<3>& VFvs)
{
    float ratio = float(VF.width())/float(VG.width());
    while(ratio < 0.5f)   // if subject resolution is substantially lower, downsample template
    {
        tipl::downsampling(VG);
        VGvs *= 2.0f;
        ratio *= 2.0f;
        tipl::out() << "ratio lower than 0.5, downsampling template to " << VGvs[0] << " mm resolution" << std::endl;
    }
    while(ratio > 2.5f)  // if subject resolution is higher, downsample it for registration
    {
        tipl::downsampling(VF);
        VFvs *= 2.0f;
        ratio /= 2.0f;
        tipl::out() << "ratio larger than 2.5, register using subject resolution of " << VFvs[0] << " mm resolution" << std::endl;
    }
}


void reconstruction_window::on_actionManual_Align_triggered()
{
    tipl::image<3> VG,VG2,VF(handle->dwi);
    tipl::vector<3> VGvs,VFvs(handle->voxel.vs);
    tipl::matrix<4,4> VG_T;
    handle->voxel.template_id = ui->primary_template->currentIndex();
    {
        if(!(tipl::io::gz_nifti(qa_template_list[handle->voxel.template_id],std::ios::in) >> VG >> VGvs >> VG_T
             >> [&](const std::string& e){QMessageBox::critical(this,"ERROR",e.c_str());}))
            return;
        tipl::io::gz_nifti(iso_template_list[handle->voxel.template_id],std::ios::in) >> VG2;
    }

    match_template_resolution(VG,VGvs,VF,VFvs);
    std::shared_ptr<manual_alignment> manual(new manual_alignment(this,
                                                                tipl::reg::template_image_pre(VG),tipl::reg::template_image_pre(VG2),VGvs,
                                                                tipl::reg::subject_image_pre(VF),tipl::reg::subject_image_pre(VF),VFvs,
                                                                tipl::reg::affine,
                                                                tipl::reg::cost_type::mutual_info));
    manual->from_T = VG_T;
    manual->to_T = handle->voxel.trans_to_mni;
    manual->on_rerun_clicked();
    if(manual->exec() != QDialog::Accepted)
        return;
    handle->voxel.qsdr_arg = manual->arg;
    handle->voxel.manual_alignment = true;
}


void reconstruction_window::on_change_fib_output_clicked()
{
    QString filename = tipl::qt::save_image_file(this,
                        windowTitle(),"FIB files (*.fz *fib.gz);;All files (*)");
    if (filename.isEmpty())
        return;
    ui->fib_output->setText(filename);
    handle->output_file_name = filename.toStdString();
}


void reconstruction_window::on_fib_output_editingFinished()
{
    handle->output_file_name = ui->fib_output->text().toStdString();
}





#include <iostream>
#include <iterator>
#include <string>
#include <cstdio>
#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include "mainwindow.h"
#include "image/image.hpp"
#include "mapping/fa_template.hpp"
#include "mapping/atlas.hpp"
#include <iostream>
#include <iterator>
#include "program_option.hpp"
#include "cmd/cnt.cpp" // Qt project cannot build cnt.cpp without adding this.

int rec(void);
int trk(void);
int src(void);
int ana(void);
int exp(void);
int atl(void);
int cnt(void);
int vis(void);


fa_template fa_template_imp;
std::vector<atlas> atlas_list;
QStringList search_files(QString dir,QString filter)
{
    QStringList dir_list,src_list;
    dir_list << dir;
    for(unsigned int i = 0;i < dir_list.size();++i)
    {
        QDir cur_dir = dir_list[i];
        QStringList new_list = cur_dir.entryList(QStringList(""),QDir::AllDirs|QDir::NoDotAndDotDot);
        for(unsigned int index = 0;index < new_list.size();++index)
            dir_list << cur_dir.absolutePath() + "/" + new_list[index];
        QStringList file_list = cur_dir.entryList(QStringList(filter),QDir::Files|QDir::NoSymLinks);
        for (unsigned int index = 0;index < file_list.size();++index)
            src_list << dir_list[i] + "/" + file_list[index];
    }
    return src_list;
}


void load_atlas(void)
{
    QDir dir = QCoreApplication::applicationDirPath()+ "/atlas";
    QStringList atlas_name_list = dir.entryList(QStringList("*.nii"),QDir::Files|QDir::NoSymLinks);
    atlas_name_list << dir.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);
    if(atlas_name_list.empty())
    {
        dir = QDir::currentPath()+ "/atlas";
        atlas_name_list = dir.entryList(QStringList("*.nii"),QDir::Files|QDir::NoSymLinks);
        atlas_name_list << dir.entryList(QStringList("*.nii.gz"),QDir::Files|QDir::NoSymLinks);
    }
    if(atlas_name_list.empty())
        return;
    atlas_list.resize(atlas_name_list.size());
    for(int index = 0;index < atlas_name_list.size();++index)
    {
        atlas_list[index].name = QFileInfo(atlas_name_list[index]).baseName().toLocal8Bit().begin();
        atlas_list[index].filename = (dir.absolutePath() + "/" + atlas_name_list[index]).toLocal8Bit().begin();
    }

}
bool load_cerebrum_mask(image::basic_image<char,3>& fp_mask)
{

    QString wm_path;
    wm_path = QCoreApplication::applicationDirPath() + "/mni_icbm152_wm_tal_nlin_asym_09a.nii.gz";
    if(!QFileInfo(wm_path).exists())
        wm_path = QDir::currentPath() + "/mni_icbm152_wm_tal_nlin_asym_09a.nii.gz";
    if(!QFileInfo(wm_path).exists())
        return false;

    gz_nifti read_wm;
    image::basic_image<float,3> wm;
    if(read_wm.load_from_file(wm_path.toStdString().c_str()))
        read_wm.toLPS(wm);
    image::basic_image<char,3> wm_mask(wm.geometry());
    for(unsigned int index = 0;index < wm_mask.size();++index)
        if(wm[index] > 0)
            wm_mask[index] = 1;

    image::matrix<4,4,float> trans,trans1,trans2;
    trans.identity();
    trans1.identity();
    trans2.identity();
    read_wm.get_image_transformation(trans.begin());
    int z = (-trans[11]-23)/trans[10]; // cut off at mni z = -22
    int y1 = (-trans[7]-10)/trans[5];
    int y2 = (-trans[7]-37)/trans[5];
    if(y2 < y1)
        std::swap(y1,y2);
    int x1 = (-trans[3]-16)/trans[0];
    int x2 = (-trans[3]+16)/trans[0];
    if(x2 < x1)
        std::swap(x1,x2);
    image::pointer_image<char,2> I1 = make_image(image::geometry<2>(wm_mask.width(),wm_mask.height()),&wm_mask[0]+z*wm.geometry().plane_size());
    image::fill_rect(I1,image::vector<2,int>(x1,y1),image::vector<2,int>(x2,y2),0);
    image::pointer_image<char,2> I2 = make_image(image::geometry<2>(wm_mask.width(),wm_mask.height()),&wm_mask[0]+(z+1)*wm.geometry().plane_size());
    image::fill_rect(I2,image::vector<2,int>(x1,y1),image::vector<2,int>(x2,y2),0);
    image::morphology::defragment(wm_mask);
    float trans1_[16] = {-1, 0, 0, 78,
                          0,-1, 0, 76,
                          0, 0, 1,-50,
                          0, 0, 0,  1};
    float trans2_[16] = {-2, 0, 0, 78,
                          0,-2, 0, 76,
                          0, 0, 2,-50,
                          0, 0, 0,  1};
    trans.inv();
    trans1 = trans*trans1_;
    trans2 = trans*trans2_;
    if(fp_mask.geometry() == image::geometry<3>(157,189,136)) // 1mm
    {
        image::resample(wm_mask,fp_mask,trans1,image::linear);
        return true;
    }
    if(fp_mask.geometry() == image::geometry<3>(79,95,69)) // 2mm
    {
        image::resample(wm_mask,fp_mask,trans2,image::linear);
        return true;
    }
    return false;
}
image::basic_image<char,3> brain_mask;
image::basic_image<float,3> mni_t1w;
bool load_brain_mask(void)
{
    if(!brain_mask.empty())
        return true;
    QString wm_path;
    wm_path = QCoreApplication::applicationDirPath() + "/mni_icbm152_t1_tal_nlin_asym_09a.nii.gz";
    if(!QFileInfo(wm_path).exists())
        wm_path = QDir::currentPath() + "/mni_icbm152_t1_tal_nlin_asym_09a.nii.gz";
    if(!QFileInfo(wm_path).exists())
        return false;

    gz_nifti read_wm;
    if(read_wm.load_from_file(wm_path.toStdString().c_str()))
        read_wm.toLPS(mni_t1w);
    brain_mask.resize(mni_t1w.geometry());
    for(unsigned int index = 0;index < mni_t1w.size();++index)
        if(mni_t1w[index] > 2000)
            brain_mask[index] = 1;
    image::morphology::smoothing(brain_mask);
    image::morphology::smoothing(brain_mask);
    image::morphology::defragment(brain_mask);
    image::morphology::negate(brain_mask);
    image::morphology::erosion(brain_mask);
    image::morphology::erosion(brain_mask);
    image::morphology::erosion(brain_mask);
    image::morphology::smoothing(brain_mask);
    image::morphology::smoothing(brain_mask);
    image::morphology::smoothing(brain_mask);
    image::morphology::defragment(brain_mask);
    image::morphology::negate(brain_mask);
    image::morphology::erosion(brain_mask);

    image::lower_threshold(mni_t1w,0);
    image::normalize(mni_t1w,1);
    return true;
}
program_option po;
int main(int ac, char *av[])
{ 
    if(ac > 2)
    {
        std::auto_ptr<QCoreApplication> cmd;
        {
            for (int i = 1; i < ac; ++i)
                if (std::string(av[i]) == std::string("--action=cnt") ||
                    std::string(av[i]) == std::string("--action=vis"))
                {
                    cmd.reset(new QApplication(ac, av));
                    std::cout << "Starting GUI-based command line interface." << std::endl;
                    break;
                }
            if(!cmd.get())
                cmd.reset(new QCoreApplication(ac, av));
        }
        cmd->setOrganizationName("LabSolver");
        cmd->setApplicationName("DSI Studio");

        try
        {
            std::cout << "DSI Studio " << __DATE__ << ", Fang-Cheng Yeh" << std::endl;

        // options for general options
            po.init(ac,av);

            if (!po.has("action") || !po.has("source"))
            {
                std::cout << "invalid command, use --help for more detail" << std::endl;
                return 1;
            }
            QDir::setCurrent(QFileInfo(po.get("action").c_str()).absolutePath());
            if(po.get("action") == std::string("rec"))
                return rec();
            if(po.get("action") == std::string("trk"))
                return trk();
            if(po.get("action") == std::string("src"))
                return src();
            if(po.get("action") == std::string("ana"))
                return ana();
            if(po.get("action") == std::string("exp"))
                return exp();
            if(po.get("action") == std::string("atl"))
                return atl();
            if(po.get("action") == std::string("cnt"))
                return cnt();
            if(po.get("action") == std::string("vis"))
                return vis();
            std::cout << "invalid command, use --help for more detail" << std::endl;
            return 1;
        }
        catch(const std::exception& e ) {
            std::cout << e.what() << std::endl;
        }
        catch(...)
        {
            std::cout << "unknown error occured" << std::endl;
        }

        return 1;
    }
    QApplication a(ac,av);
    a.setOrganizationName("LabSolver");
    a.setApplicationName("DSI Studio");
    QFont font;
    font.setFamily(QString::fromUtf8("Arial"));
    a.setFont(font);
    // load template
    if(!fa_template_imp.load_from_file())
    {
        QMessageBox::information(0,"Error","Cannot find HCP488_QA.nii.gz in file directory",0);
        return false;
    }
    // load atlas
    load_atlas();

    MainWindow w;
    w.setFont(font);
    w.show();
    w.setWindowTitle(QString("DSI Studio ") + __DATE__ + " build");
    return a.exec();
}

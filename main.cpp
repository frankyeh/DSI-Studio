#include <iostream>
#include <iterator>
#include <string>
#include <QApplication>
#include <QCleanlooksStyle>
#include <QMetaObject>
#include <QMetaMethod>
#include <QMessageBox>
#include "mainwindow.h"
#include "mat_file.hpp"
#include "boost/program_options.hpp"
#include "image/image.hpp"
#include "mapping/fa_template.hpp"
namespace po = boost::program_options;

int rec(int ac, char *av[]);
int trk(int ac, char *av[]);
int src(int ac, char *av[]);

fa_template fa_template_imp;

void prog_debug(const char* file,const char* fun)
{

}
struct QSignalSpyCallbackSet
{
    typedef void (*BeginCallback)(QObject *caller, int method_index, void **argv);
    typedef void (*EndCallback)(QObject *caller, int method_index);
    BeginCallback signal_begin_callback;
    BeginCallback slot_begin_callback;
    EndCallback signal_end_callback;
    EndCallback slot_end_callback;
};
void Q_CORE_EXPORT qt_register_signal_spy_callbacks(const QSignalSpyCallbackSet &callback_set);
extern QSignalSpyCallbackSet Q_CORE_EXPORT qt_signal_spy_callback_set;


const QString q4pugss_value(const char *type, void *argv)
{
        QVariant v( QVariant::nameToType(type), argv );
        if( v.type() )
                return QString("%1(%2)").arg(type).arg(v.toString());
        return QString("%1 <cannot decode>").arg(type);
}

bool q4pugss_GetMethodString(QObject* caller, int method_index, void **argv,  QString &string)
{
        const QMetaObject * mo = caller->metaObject();
        if( !mo )
                return false;
        QMetaMethod m = mo->method(method_index);
        if( method_index >= mo->methodCount() )
                return false;
        if(QString(m.signature()) == QString("awake()") ||
           QString(m.signature()) == QString("aboutToBlock()"))
            return false;

        static QString methodType[] = {"Method", "Signal", "Slot"};

        string = QString("%1 (%2) ")
                                .arg(caller->objectName().isNull()?"noname":caller->objectName())
                                .arg(mo->className());
        string += QString("%1: %2(")
                                .arg(methodType[(int)m.methodType()])
                                .arg(QString(m.signature()).section('(',0,0));

        QList<QByteArray> pNames = m.parameterNames();
        QList<QByteArray> pTypes = m.parameterTypes();

        for(int i=0; i<pNames.count(); i++) {
                string += QString("%1=%2")
                                        .arg(QString(pNames.at(i)))
                                        .arg(q4pugss_value(pTypes.at(i), argv[i+1]));
                if(i != pNames.count()-1)
                        string += ", ";
        }

        string += QString(")");

        return true;
}
void q4pugss_BeginCallBackSignal(QObject* caller, int method_index, void **argv)
{
    QString sig_param;
    if(q4pugss_GetMethodString(caller, method_index, argv, sig_param))
        std::cout << (const char*)(sig_param.toLocal8Bit()) << std::endl;
}

void q4pugss_EndCallBackSignal(QObject*, int)
{
}

void q4pugss_BeginCallBackSlot(QObject* caller, int method_index, void **argv)
{
    QString sig_param;
    if(q4pugss_GetMethodString(caller, method_index, argv, sig_param))
        std::cout << (const char*)(sig_param.toLocal8Bit()) << std::endl;
}


void q4pugss_EndCallBackSlot(QObject*, int)
{
}

#include "mapping/normalization.hpp"
#include "mapping/mni_norm.hpp"
#include <iostream>
#include <iterator>
std::string program_base;
int main(int ac, char *av[])
{

    /*
    Results without SPM_DEBUG
    0.415798,0.00118842,0.00536883,10.4873
    0.0121313,0.407114,0.084006,2.66877
    -0.0065242,-0.0990291,0.404675,2.75814
     FWHM = 5.34324 Var = 0.0818234
     FWHM = 4.51921 Var = 0.048115
     FWHM = 4.34149 Var = 0.043301
     FWHM = 4.22131 Var = 0.0400756
     FWHM = 4.13327 Var = 0.0375324
     FWHM = 4.06334 Var = 0.0357627
     FWHM = 4.01697 Var = 0.0347199
     FWHM = 3.98543 Var = 0.0338868
     FWHM = 3.95097 Var = 0.0331734
     FWHM = 3.92201 Var = 0.0325709
     FWHM = 3.8912 Var = 0.0320772
     FWHM = 3.86547 Var = 0.0316088
     FWHM = 3.84749 Var = 0.0312694
     FWHM = 3.83566 Var = 0.0310322
     FWHM = 3.82815 Var = 0.0308409
     FWHM = 3.82004 Var = 0.0306298
        13.4767 6.10598 8.46129
    */
    /*
    normalization<image::basic_image<double,3> > n;
    if(!n.load_from_file("FMRIB58_FA_1mm.nii","GFA_0171.src.gz.odf8.f5rec.gqi.1.3.fib.nii"))
        return 0;
    n.normalize();
    image::basic_image<double,3> out;



    n.warp_image(n.VF,out,2.0);
    image::vector<3,double> p1,p2;
    n.warp_coordinate(p1,p2);
    std::cout<< p2 << std::endl;
    image::io::nifti header;
    #ifdef SPM_DEBUG
    image::flip_x(out);
    #else
    image::flip_xy(out);
    #endif
    header << out;
    header.set_image_transformation(n.trans_to_mni);
    header.save_to_file("c:/warp.nii");
    return 0;
    */
    program_base = av[0];
    if(ac > 2)
    {
        {
            std::cout << "DSI Studio " << __DATE__ << ", Fang-Cheng Yeh" << std::endl;

        // options for general options
            po::options_description desc("reconstruction options");
            desc.add_options()
            ("help", "help message")
            ("action", po::value<std::string>(), "rec:diffusion reconstruction trk:fiber tracking")
            ("source", po::value<std::string>(), "assign the .src or .fib file name")
            ;


            po::variables_map vm;
            po::store(po::command_line_parser(ac, av).options(desc).allow_unregistered().run(),vm);
            if (vm.count("help"))
            {
                std::cout << "example: perform reconstruction" << std::endl;
                std::cout << "    --action=rec --source=test.src.gz --method=4 " << std::endl;
                std::cout << "options:" << std::endl;
                rec(0,0);
                std::cout << "example: perform fiber tracking" << std::endl;
                std::cout << "    --action=trk --source=test.src.gz.fib.gz --method=0 --fiber_count=5000" << std::endl;
                std::cout << "options:" << std::endl;
                trk(0,0);
                return 1;
            }

            if (!vm.count("action") || !vm.count("source"))
            {
                std::cout << "invalid command, use --help for more detail" << std::endl;
                return 1;
            }
            if(vm["action"].as<std::string>() == std::string("rec"))
                return rec(ac,av);
            if(vm["action"].as<std::string>() == std::string("trk"))
                return trk(ac,av);
            if(vm["action"].as<std::string>() == std::string("src"))
                return src(ac,av);
        }
        return 1;
    }


    /*
    QSignalSpyCallbackSet cb;
    cb.signal_begin_callback = q4pugss_BeginCallBackSignal;
    cb.signal_end_callback   = q4pugss_EndCallBackSignal;
    cb.slot_begin_callback   = q4pugss_BeginCallBackSlot;
    cb.slot_end_callback     = q4pugss_EndCallBackSlot;
    qt_register_signal_spy_callbacks(cb);
    */

    QApplication::setStyle(new QCleanlooksStyle);
    QApplication a(ac,av);
    a.setOrganizationName("LabSolver");
    a.setApplicationName("DSI Studio");
    QFont font;
    font.setFamily(QString::fromUtf8("Arial"));
    a.setFont(font);

    {
        int pos = 0;
        for(int index = 0;av[0][index];++index)
            if(av[0][index] == '\\' || av[0][index] == '/')
                pos = index;
        std::string fa_template_path(&(av[0][0]),&(av[0][0])+pos+1);
        fa_template_path += "FMRIB58_FA_1mm.nii";
        if(!fa_template_imp.load_from_file(fa_template_path.c_str()))
        {
            QMessageBox::information(0,"Error","Cannot find the fa template file",0);
            return 0;
        }
    }
    MainWindow w;
    w.setFont(font);
    w.showMaximized();
    w.setWindowTitle(QString("DSI Studio ") + __DATE__ + " build");
    return a.exec();
}

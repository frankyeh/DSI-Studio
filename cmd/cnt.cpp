#include <QApplication>
#include <QFileInfo>
#include "tracking/region/Regions.h"
#include "tracking/atlasdialog.h"
#include "tracking/roi.hpp"
#include "connectometry/group_connectometry_analysis.h"
bool load_roi(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,std::shared_ptr<RoiMgr> roi_mgr);
int cnt(tipl::program_option<tipl::out>& po)
{
    auto vbc = std::make_shared<group_connectometry_analysis>();
    if(!vbc->load_database(po.get("source").c_str()))
        return tipl::error() << vbc->error_msg,1;

    auto& db = vbc->handle->db;
    if(db.demo.empty() || po.has("demo"))
    {
        // read demographic file
        if(!po.check("demo"))
            return 1;
        if(!db.parse_demo(po.get("demo")))
            return tipl::error() << vbc->handle->error_msg,1;
    }

    tipl::out() << "available index name: " << tipl::merge(db.index_list,',');
    if(!db.set_current_index(po.get("index_name",db.index_list.front())))
        return tipl::error() << "cannot find " << po.get("index_name")
                             << " in the database",1;

    {
        std::string sout("selectable variables include ");
        for(size_t i = 0;i < db.feature.size();++i)
             sout += "\t(" + std::to_string(i) + ")" +db.feature[i].title;
        tipl::out() << sout;
    }

    if(!po.check("voi") || !po.check("variable_list"))
        return 1;

    // shared with the GUI's "set_voi" command (connectometry_db::select_voi), which also accepts
    // feature names instead of indices; --voi/--variable_list here still take the documented indices
    std::string foi_str = db.select_voi(po.get("voi"),po.get("variable_list"));
    if(foi_str.empty())
        return tipl::error() << vbc->handle->error_msg,1;

    {
        tipl::progress prog("connectometry parameters");
        vbc->no_tractogram = po.get("no_tractogram",1);
        if(!tipl::show_prog && vbc->no_tractogram == 0)
        {
            tipl::warning() << "cannot generate tractogram at command line mode. no_tractogram is disabled" ;
            vbc->no_tractogram = 1;
        }
        vbc->region_pruning = (po.get("region_pruning",1) == 1);
        if(db.type == connectometry_db::longitudinal_type::plain)
            vbc->normalize_iso = (po.get("normalize_iso",1) == 1);
        vbc->foi_str = foi_str;
        vbc->length_threshold_voxels = po.get("length_threshold",(vbc->handle->dim[0]/4)/5*5);
        vbc->tip_iteration = po.get("tip_iteration",16);
        vbc->fdr_threshold = po.get("fdr_threshold",0.0f);

        // select cohort and feature
        vbc->model = std::make_shared<stat_model>();
        vbc->model->read_demo(db);
        if(!vbc->model->select_cohort(db,po.get("select")) || !vbc->model->select_feature(db,vbc->foi_str))
            return tipl::error() << vbc->model->error_msg,1;
        size_t n = std::count(vbc->model->remove_list.begin(),
                              vbc->model->remove_list.end(),false);
        tipl::out() << "sample size:" << n;
        if(n <= 2)
            return tipl::error() << "not enough sample size: " << n,1;


        if(po.has("t_threshold"))
        {
            auto t = vbc->t_threshold = po.get("t_threshold",2.5f);
            vbc->rho_threshold = t/std::sqrt(t*t+n-2);
        }
        else
        {
            auto rho = vbc->rho_threshold = po.get("effect_size",0.3f);
            vbc->t_threshold = rho*std::sqrt(double(n)-2)/std::sqrt(1-rho*rho);
        }

        // setup roi
        vbc->roi_mgr = std::make_shared<RoiMgr>(vbc->handle);
        if(po.get("exclude_cb",0))
            vbc->exclude_cerebellum();

        if(!load_roi(po,vbc->handle,vbc->roi_mgr))
            return 1;

        // if no seed assigned, assign whole brain
        if(vbc->roi_mgr->seeds.empty())
            vbc->roi_mgr->setWholeBrainSeed(vbc->fiber_threshold);

    }

    {
        tipl::progress prog("running connectometry");
        if(po.has("output"))
            vbc->output_file_name = po.get("output",std::string());
        vbc->run_permutation(tipl::max_thread_count,po.get("permutation",uint32_t(2000)));
        vbc->wait(0);
        vbc->generate_report();
    }
    return 0;
}

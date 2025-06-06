#include <QApplication>
#include <QFileInfo>
#include "tracking/region/Regions.h"
#include "tracking/atlasdialog.h"
#include "tracking/roi.hpp"
#include "connectometry/group_connectometry_analysis.h"
bool load_roi(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,std::shared_ptr<RoiMgr> roi_mgr);
int cnt(tipl::program_option<tipl::out>& po)
{
    std::shared_ptr<group_connectometry_analysis> vbc(new group_connectometry_analysis);
    if(!vbc->load_database(po.get("source").c_str()))
    {
        tipl::error() << vbc->error_msg << std::endl;
        return 1;
    }

    auto& db = vbc->handle->db;
    if(db.demo.empty() || po.has("demo"))
    {
        // read demographic file
        if(!po.check("demo"))
            return 1;
        if(!db.parse_demo(po.get("demo")))
        {
            tipl::error() << vbc->handle->error_msg << std::endl;
            return 1;
        }
    }

    tipl::out() << "available index name: " << tipl::merge(db.index_list,',');
    if(!db.set_current_index(po.get("index_name",db.index_list.front())))
    {
        tipl::error() << "cannot find " << po.get("index_name") << " in the database";
        return 1;
    }

    if(!po.check("voi") || !po.check("variable_list"))
        return 1;


    {
        std::string sout("selectable variables include ");
        for(size_t i = 0;i < db.feature_titles.size();++i)
             sout += "\t(" + std::to_string(i) + ")" +db.feature_titles[i];
        tipl::out() << sout;
    }

    std::vector<unsigned int> variable_list;
    std::string foi_str;


    {

        std::string var_text = po.get("variable_list");
        std::replace(var_text.begin(),var_text.end(),',',' ');
        std::istringstream var_in(var_text);
        variable_list.assign((std::istream_iterator<int>(var_in)),(std::istream_iterator<int>()));
        for(auto v : variable_list)
            if(v >= db.feature_titles.size())
            {
                tipl::error() << "invalid variable value: " << v << std::endl;
                return 1;
            }


        if(po.get("voi") == "Intercept" | po.get("voi") == "longitudinal")
        {
            if(!db.is_longitudinal)
            {
                tipl::error() << "The longitudinal change can only be studied in a longitudinal database." << std::endl;
                return 1;
            }
            foi_str = "longitudinal change";
        }
        else
        {
            unsigned int voi_index = po.get("voi",variable_list.front());
            if(voi_index >= db.feature_titles.size())
            {
                tipl::error() << "invalid variable of interest: " << voi_index << std::endl;
                return 1;
            }
            // the variable to study needs to be included in the model
            variable_list.push_back(voi_index);
            foi_str = db.feature_titles[voi_index];
        }

        {
            // sort and variables, make them unique
            std::set<unsigned int> s(variable_list.begin(),variable_list.end());
            variable_list.assign(s.begin(),s.end());
        }

        std::fill(db.feature_selected.begin(),db.feature_selected.end(),false);
        for(auto index : variable_list)
            db.feature_selected[index] = true;
    }

    {
        tipl::progress prog("connectometry parameters");
        vbc->no_tractogram = (po.get("no_tractogram",1) == 1);
        vbc->region_pruning = (po.get("region_pruning",1) == 1);
        if(!db.is_longitudinal)
            vbc->normalize_iso = (po.get("normalize_iso",1) == 1);
        vbc->foi_str = foi_str;
        vbc->length_threshold_voxels = po.get("length_threshold",(vbc->handle->dim[0]/4)/5*5);
        vbc->tip_iteration = po.get("tip_iteration",16);
        vbc->fdr_threshold = po.get("fdr_threshold",0.0f);

        // select cohort and feature
        vbc->model.reset(new stat_model);
        vbc->model->read_demo(db);
        if(!vbc->model->select_cohort(db,po.get("select")) || !vbc->model->select_feature(db,vbc->foi_str))
        {
            tipl::error() << vbc->model->error_msg.c_str() << std::endl;
            return 1;
        }
        size_t n = 0;
        for(size_t i = 0;i < vbc->model->remove_list.size();++i)
        {
            if(!vbc->model->remove_list[i])
                n++;
        }
        tipl::out() << "sample size:" << n;
        if(n <= 2)
        {
            tipl::error() << "not enough sample size: " << n;
            return 1;
        }


        if(po.has("t_threshold"))
        {
            auto t = vbc->t_threshold = po.get("t_threshold",2.5f);
            vbc->rho_threshold = t/std::sqrt(t*t+n-2);
        }
        else
        {
            auto rho = vbc->rho_threshold = po.get("effect_size",0.3f);
            vbc->t_threshold = rho*std::sqrt(double(n)-2)/(1-rho*rho);
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
        for(auto& thread: vbc->threads)
            if(thread.joinable())
                thread.join();
    }
    vbc->save_result();
    vbc->calculate_FDR();
    std::string output;
    vbc->generate_report(output);
    {
        std::string report_file_name = vbc->output_file_name+".report.html";
        std::ofstream out(report_file_name.c_str());
        if(!out)
        {
            tipl::error() << "cannot output file to " << report_file_name << std::endl;
            return 1;
        }
        else
        {
            tipl::out() << "saving " << report_file_name;
            out << output << std::endl;
        }
    }
    return 0;
}

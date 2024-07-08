#include <QFileInfo>
#include <QStringList>
#include <QDir>
#include <iostream>
#include <iterator>
#include <string>
#include <filesystem>
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "mapping/atlas.hpp"
#include "SliceModel.h"
#include "connectometry/group_connectometry_analysis.h"


extern std::vector<std::shared_ptr<CustomSliceModel> > other_slices;

std::shared_ptr<CustomSliceModel> load_slices(std::shared_ptr<fib_data> handle,std::string file_name)
{
    auto new_slice = std::make_shared<CustomSliceModel>(handle.get());
    if(!new_slice->load_slices(file_name))
    {
        tipl::error() << "fail to load " << file_name << " " << new_slice->error_msg << std::endl;
        return std::shared_ptr<CustomSliceModel>();
    }
    new_slice->wait();
    new_slice->update_transform();
    tipl::out() << "size: " << new_slice->dim << "resolution: " << new_slice->vs << std::endl;
    tipl::out() << "srow:";
    tipl::out() << new_slice->trans_to_mni;
    tipl::out() << "to_dif:";
    tipl::out() << new_slice->to_dif;
    return new_slice;
}

bool check_other_slices(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle)
{
    if(!other_slices.empty() || !po.has("other_slices"))
        return true;
    std::vector<std::string> filenames;
    if(!po.get_files("other_slices",filenames))
    {
        tipl::error() << po.error_msg << std::endl;
        return false;
    }
    for(const auto& each : filenames)
    {
        tipl::out() << "add slice: " << each << std::endl;
        auto new_slice = load_slices(handle,each);
        if(!new_slice.get())
            return false;
        other_slices.push_back(new_slice);
    }
    return true;
}
bool export_track_info(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,
                       std::string file_name,
                       std::shared_ptr<TractModel> tract_model)
{
    std::istringstream in(po.get("export"));
    std::string cmd;
    while(std::getline(in,cmd,','))
    {
        // track analysis report
        if(cmd.find("report") == 0)
        {
            tipl::out() << "export track analysis report..." << std::endl;
            std::replace(cmd.begin(),cmd.end(),':',' ');
            std::istringstream in(cmd);
            std::string report_tag,index_name;
            uint32_t profile_dir = 0,bandwidth = 0;
            in >> report_tag >> index_name >> profile_dir >> bandwidth;
            std::vector<float> values,data_profile,data_ci1,data_ci2;
            // check index
            if(index_name != "qa" && index_name != "fa" &&  handle->get_name_index(index_name) == handle->view_item.size())
            {
                tipl::out() << "cannot find index name " << index_name << std::endl;
                return false;
            }
            if(bandwidth == 0)
            {
                tipl::out() << "please specify bandwidth value" << std::endl;
                return false;
            }
            if(profile_dir > 4)
            {
                tipl::out() << "please specify a valid profile type" << std::endl;
                return false;
            }
            tipl::out() << "calculating report" << std::endl;
            tipl::out() << "profile_dir: " << profile_dir << std::endl;
            tipl::out() << "bandwidth: " << bandwidth << std::endl;
            tipl::out() << "index_name: " << index_name << std::endl;
            tract_model->get_report(
                                handle,
                                profile_dir,
                                bandwidth,
                                index_name,
                                values,data_profile,data_ci1,data_ci2);

            std::replace(cmd.begin(),cmd.end(),' ','.');
            std::string file_name_stat = file_name + "." + cmd + ".txt";
            tipl::out() << "saving " << file_name_stat << std::endl;
            std::ofstream report(file_name_stat.c_str());
            report << "position\t";
            std::copy(values.begin(),values.end(),std::ostream_iterator<float>(report,"\t"));
            report << std::endl;
            report << "value\t";
            std::copy(data_profile.begin(),data_profile.end(),std::ostream_iterator<float>(report,"\t"));
            if(!data_ci1.empty())
            {
                report << std::endl;
                report << "CI\t";
                std::copy(data_ci1.begin(),data_ci1.end(),std::ostream_iterator<float>(report,"\t"));
            }
            if(!data_ci2.empty())
            {
                report << std::endl;
                report << "CI\t";
                std::copy(data_ci2.begin(),data_ci2.end(),std::ostream_iterator<float>(report,"\t"));
            }
            report << std::endl;
            continue;
        }

        std::string file_name_stat = file_name + "." + cmd;
        // export statistics
        if(QString(cmd.c_str()).startsWith("tdi"))
        {
            float ratio = 1.0f;
            {
                std::string digit(cmd);
                digit.erase(std::remove_if(digit.begin(),digit.end(),[](char ch){return (ch < '0' || ch > '9') && ch != '.';}),digit.end());
                if(!digit.empty())
                    (std::istringstream(digit)) >> ratio;
                tipl::out() << "calculating TDI at x" << ratio << " resolution" << std::endl;
            }

            bool output_color = QString(cmd.c_str()).contains("color");
            bool output_end = QString(cmd.c_str()).contains("end");
            file_name_stat += ".nii.gz";
            tipl::matrix<4,4> to_t1t2,trans_to_mni;
            tipl::shape<3> dim;
            tipl::vector<3,float> vs;
            to_t1t2.identity();
            dim = handle->dim;
            vs = handle->vs;

            if(po.has("ref"))
            {
                auto new_slice = load_slices(handle,po.get("ref"));
                if(!new_slice.get())
                    return false;
                dim = new_slice->dim;
                vs = new_slice->vs;
                trans_to_mni = new_slice->trans_to_mni;
                to_t1t2 = new_slice->to_slice;
            }
            else
            {
                if(ratio != 1.0f)
                {
                    to_t1t2[0] = to_t1t2[5] = to_t1t2[10] = ratio;
                    dim = handle->dim*ratio;
                    vs /= ratio;
                }
            }
            std::vector<std::shared_ptr<TractModel> > tract;
            tract.push_back(tract_model);
            if(output_color)
                tipl::out() << " in RGB color";
            if(output_end)
                tipl::out() << " end point only";
            tipl::out() << std::endl;
            tipl::out() << "TDI dimension: " << dim << std::endl;
            tipl::out() << "TDI voxel size: " << vs << std::endl;
            tipl::out() << std::endl;
            tipl::out() << "saving " << file_name_stat;
            if(!TractModel::export_tdi(file_name_stat.c_str(),tract,dim,vs,trans_to_mni,to_t1t2,output_color,output_end))
            {
                tipl::error() << "failed to save file. Please check write permission." << std::endl;
                return false;
            }
            continue;
        }


        if(cmd == "stat")
        {
            file_name_stat += ".txt";
            tipl::out() << "saving " << file_name_stat << std::endl;
            std::ofstream out_stat(file_name_stat.c_str());
            if(!out_stat)
            {
                tipl::out() << "Output statistics to file_name_stat failed. Please check write permission" << std::endl;
                return false;
            }
            std::string result;
            tract_model->get_quantitative_info(handle,result);
            out_stat << result;
            continue;
        }

        {
            if(cmd.find('.') != std::string::npos)
                cmd = cmd.substr(0,cmd.find('.'));
            else
                file_name_stat += ".txt";
            if(handle->get_name_index(cmd) != handle->view_item.size())
            {
                tract_model->save_data_to_file(handle,file_name_stat.c_str(),cmd);
                continue;
            }
        }
        tipl::out() << "invalid export option " << cmd << std::endl;
        return false;
    }
    return true;
}

bool load_nii(tipl::program_option<tipl::out>& po,
              std::shared_ptr<fib_data> handle,
              const std::string& file_name,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::vector<std::string>& names);

bool get_connectivity_matrix(tipl::program_option<tipl::out>& po,
                             std::shared_ptr<fib_data> handle,
                             std::string output_name,
                             std::shared_ptr<TractModel> tract_model)
{
    QStringList connectivity_list = QString(po.get("connectivity").c_str()).split(",");
    QStringList connectivity_type_list = QString(po.get("connectivity_type","pass").c_str()).split(",");
    QStringList connectivity_value_list = QString(po.get("connectivity_value","count").c_str()).split(",");
    std::string connectivity_output = po.get("connectivity_output","matrix,connectogram,measure");
    for(int i = 0;i < connectivity_list.size();++i)
    {
        std::string roi_file_name = connectivity_list[i].toStdString();
        tipl::out() << "opening " << roi_file_name << std::endl;
        ConnectivityMatrix data;

        // specify atlas name (e.g. --connectivity=AAL2)
        if(!QString(roi_file_name.c_str()).contains("."))
        {
            auto at = handle->get_atlas(roi_file_name);
            if(!at.get())
            {
                tipl::error() << handle->error_msg << std::endl;
                return false;
            }
            data.set_atlas(at,handle);
        }
        else
        // specify atlas file (e.g. --connectivity=subject_file.nii.gz)
        {
            if(!std::filesystem::exists(roi_file_name))
            {
                tipl::error() << "cannot open file " << roi_file_name << std::endl;
                return false;
            }

            if(QString(roi_file_name.c_str()).toLower().endsWith("txt")) // a roi list
            {
                tipl::out() << "reading " << roi_file_name << " as an ROI list" << std::endl;
                std::string dir = QFileInfo(roi_file_name.c_str()).absolutePath().toStdString();
                dir += "/";
                std::ifstream in(roi_file_name.c_str());
                std::string line;
                std::vector<std::shared_ptr<ROIRegion> > regions;
                while(std::getline(in,line))
                {
                    std::shared_ptr<ROIRegion> region(new ROIRegion(handle));
                    std::string fn;
                    if(std::filesystem::exists(line))
                        fn = line;
                    else
                        fn = dir + line;
                    if(!region->load_region_from_file(fn.c_str()))
                    {
                        tipl::error() << "failed to open file as a region: " << fn << std::endl;
                        return false;
                    }
                    regions.push_back(region);
                    data.region_name.push_back(QFileInfo(line.c_str()).baseName().toStdString());
                }
                data.set_regions(handle->dim,regions);
                tipl::out() << "a total of " << data.region_count << " regions are loaded." << std::endl;
            }
            else
            {
                tipl::out() << "reading " << roi_file_name << " as a NIFTI ROI file" << std::endl;
                std::vector<std::shared_ptr<ROIRegion> > regions;
                std::vector<std::string> names;
                if(!load_nii(po,handle,roi_file_name,regions,names))
                    return false;
                data.region_name = names;
                data.set_regions(handle->dim,regions);
            }
        }
        for(int j = 0;j < connectivity_type_list.size();++j)
        for(int k = 0;k < connectivity_value_list.size();++k)
        {
            std::string connectivity_roi = roi_file_name;
            std::string connectivity_value = connectivity_value_list[k].toStdString();
            bool use_end_only = connectivity_type_list[j].toLower() == QString("end");
            QDir pwd = QDir::current();
            if(connectivity_value == "trk")
                QDir::setCurrent(QFileInfo(output_name.c_str()).absolutePath());
            if(!data.calculate(handle,*(tract_model.get()),
                               connectivity_value,
                               use_end_only,po.get("connectivity_threshold",0.001f)))
            {
                tipl::error() << data.error_msg << std::endl;
                return false;
            }
            if(data.overlap_ratio > 0.5f)
            {
                tipl::out() << "the ROIs have a large overlapping area (ratio: "
                          << data.overlap_ratio << "). The network measure calculated may not be reliable" << std::endl;
            }
            if(connectivity_value == "trk")
            {
                // restore previous working directory
                QDir::setCurrent(pwd.path());
                continue;
            }

            std::string file_name_stat(output_name);
            file_name_stat += ".";
            file_name_stat += (std::filesystem::exists(connectivity_roi)) ? QFileInfo(connectivity_roi.c_str()).baseName().toStdString():connectivity_roi;
            file_name_stat += ".";
            file_name_stat += connectivity_value;
            file_name_stat += use_end_only ? ".end":".pass";

            if(connectivity_output.find("matrix") != std::string::npos)
            {
                std::string matrix = file_name_stat + ".connectivity.mat";
                tipl::out() << "saving " << matrix << std::endl;
                data.save_to_file(matrix.c_str());
            }

            if(connectivity_output.find("connectogram") != std::string::npos)
            {
                std::string connectogram = file_name_stat + ".connectogram.txt";
                tipl::out() << "saving " << connectogram << std::endl;
                data.save_to_connectogram(connectogram.c_str());
            }

            if(connectivity_output.find("measure") != std::string::npos)
            {
                std::string measure = file_name_stat + ".network_measures.txt";
                tipl::out() << "saving " << measure << std::endl;
                std::string report;
                data.network_property(report);
                std::ofstream out(measure.c_str());
                out << report;
            }
        }
    }
    return true;
}

// test example
// --action=trk --source=./test/20100129_F026Y_WANFANGYUN.src.gz.odf8.f3rec.de0.dti.fib.gz --method=0 --fiber_count=5000
extern std::vector<std::string> fib_template_list;
std::shared_ptr<fib_data> cmd_load_fib(std::string file_name)
{
    std::shared_ptr<fib_data> handle(new fib_data);
    if(file_name.length() == 1 && file_name[0] >= '0' && file_name[0] <= '5')
        file_name = fib_template_list[file_name[0]-'0'];
    if(!std::filesystem::exists(file_name))
    {
        tipl::error() << file_name << " does not exist. terminating..." << std::endl;
        return std::shared_ptr<fib_data>();
    }
    if (!handle->load_from_file(file_name.c_str()))
    {
        tipl::error() << handle->error_msg << std::endl;
        return std::shared_ptr<fib_data>();
    }
    return handle;
}
std::shared_ptr<fib_data> cmd_load_fib(tipl::program_option<tipl::out>& po)
{
    auto handle = cmd_load_fib(po.get("source"));
    if(!handle.get() || !check_other_slices(po,handle))
        return std::shared_ptr<fib_data>();
    return handle;
}
bool load_region(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,const std::string& region_text)
{
    QStringList str_list = QString(region_text.c_str()).split(",");// splitting actions
    std::string file_name = str_list[0].toStdString();
    std::string region_name;

    // --roi=file_name:value but avoid windows path that includes drive letter
    {
        auto pos = file_name.find_last_of(':');
        if(pos != std::string::npos && pos != 1) // Windows path
        {
            region_name = file_name.substr(pos+1);
            file_name = file_name.substr(0,pos);
        }
    }
    tipl::out() << "load " << (region_name.empty() ? std::string("volume"):region_name) << " from " << file_name << std::endl;

    if(QString(file_name.c_str()).endsWith(".nii.gz") ||
       QString(file_name.c_str()).endsWith(".nii"))
    {
        std::vector<std::shared_ptr<ROIRegion> > regions;
        std::vector<std::string> names;
        if(!load_nii(po,handle,file_name,regions,names))
            return false;
        if(region_name.empty())
            roi = *(regions[0].get());
        else
        {
            bool found = false;
            for(size_t index = 0;index < names.size();++index)
                if(names[index] == region_name ||
                   names[index] == QFileInfo(file_name.c_str()).baseName().toStdString() + "_" + region_name)
                {
                    found = true;
                    roi = *(regions[index].get());
                    break;
                }
            if(!found)
            {
                tipl::error() << "cannot find " << region_name << " in the NIFTI file." << std::endl;
                return false;
            }
        }
    }
    else
    {
        if(!region_name.empty())
        {
            std::vector<tipl::vector<3,short> > points;
            if(!handle->get_atlas_roi(file_name,region_name,points))
            {
                tipl::error() << handle->error_msg << std::endl;
                return false;
            }
            roi.add_points(std::move(points));
        }
        else
            if(!roi.load_region_from_file(file_name.c_str()))
            {
                tipl::error() << "cannot open file as a region" << file_name << std::endl;
                return false;
            }
    }

    // now perform actions
    for(int i = 1;i < str_list.size();++i)
    {
        tipl::out() << str_list[i].toStdString() << " applied." << std::endl;
        roi.perform(str_list[i].toStdString());
    }
    if(roi.region.empty())
        tipl::warning() << file_name << " is an empty region file" << std::endl;

    return true;
}

int trk_post(tipl::program_option<tipl::out>& po,
             std::shared_ptr<fib_data> handle,
             std::shared_ptr<TractModel> tract_model,
             std::string tract_file_name,bool output_track)
{
    tipl::progress prog("post-tracking analysis");

    auto tip_iteration = uint8_t(po.get("tip_iteration", (po.has("track_id") | po.has("dt_metric1") ) ? 4 : 0));
    if(tip_iteration)
    {
        for(size_t i = 0;i < tip_iteration &&
            tract_model->get_visible_track_count() &&
            tract_model->trim();++i)
            ;
        tipl::out() << tract_model->get_deleted_track_count() << " tracts are removed by pruning." << std::endl;
        tipl::out() << "Total tract count after pruning is " << tract_model->get_visible_track_count() << " tracts." << std::endl;
    }
    if(tract_model->get_visible_track_count() == 0)
    {
        tipl::out() << "No tract for further processing" << std::endl;
        return 0;
    }
    if (po.has("delete_repeat"))
    {
        tipl::out() << "deleting repeat tracks..." << std::endl;
        float distance = po.get("delete_repeat",float(1));
        tract_model->delete_repeated(distance);
        tipl::out() << "repeat tracks with distance smaller than " << distance <<" voxel distance are deleted" << std::endl;
    }
    if(po.has("cluster"))
    {
        std::string cmd = po.get("cluster");
        std::replace(cmd.begin(),cmd.end(),',',' ');
        std::istringstream in(cmd);
        int method = 0,count = 0,detail = 0;
        std::string name;
        in >> method >> count >> detail >> name;
        tipl::out() << "cluster method: " << method << std::endl;
        tipl::out() << "cluster count: " << count << std::endl;
        tipl::out() << "cluster resolution (if method is 0) : " << detail << " mm" << std::endl;
        tipl::out() << "run clustering." << std::endl;
        tract_model->run_clustering(uint8_t(method),uint32_t(count),detail);
        tipl::out() << "saving " << tract_file_name << "." << name << std::endl;
        std::ofstream out(tract_file_name + "." + name);
        std::copy(tract_model->tract_cluster.begin(),tract_model->tract_cluster.end(),std::ostream_iterator<int>(out," "));
    }
    if(po.has("recognize"))
    {
        std::vector<unsigned int> labels;
        std::vector<std::string> names;
        handle->recognize(tract_model,labels,names);
        tipl::out() << "saving " << (po.get("recognize") + ".label.txt") << std::endl;
        std::ofstream out1(po.get("recognize") + ".label.txt");
        std::copy(labels.begin(),labels.end(),std::ostream_iterator<int>(out1," "));

        tipl::out() << "saving " << (po.get("recognize") + ".name.txt") << std::endl;
        std::ofstream out2(po.get("recognize") + ".name.txt");
        std::copy(names.begin(),names.end(),std::ostream_iterator<std::string>(out2," "));
        tract_model->tract_cluster = labels;
    }
    if(output_track)
    {
        bool failed = false;
        if(po.has("ref")) // save track in T1W/T2W space
        {
            auto new_slice = load_slices(handle,po.get("ref"));
            if(!new_slice.get())
                return 1;
            if(!tract_model->save_transformed_tracts_to_file(tract_file_name.c_str(),new_slice->dim,new_slice->vs,new_slice->trans_to_mni,new_slice->to_slice,false))
                failed = true;
        }
        else
        if (!tract_model->save_tracts_to_file(tract_file_name.c_str()))
            failed = true;

        if(failed)
        {
            tipl::error() << "cannot save tracks as " << tract_file_name << ". Please check write permission, directory, and disk space." << std::endl;
            return 1;
        }
    }


    if(po.has(("native_track")) && !tract_model->save_tracts_in_native_space(handle,po.get("native_track").c_str()))
    {
        tipl::error() << "failed to save file to " << po.get("native_track") << std::endl;
        return 1;
    }
    if(po.has(("template_track")) && !tract_model->save_tracts_in_template_space(handle,po.get("template_track").c_str()))
    {
        tipl::error() << "failed to save file to " << po.get("template_track") << std::endl;
        return 1;
    }
    if(po.has(("mni_track")) && !tract_model->save_tracts_in_template_space(handle,po.get("mni_track").c_str(),true))
    {
        tipl::error() << "failed to save file to " << po.get("mni_track") << std::endl;
        return 1;
    }
    if(po.has(("end_point")) && !tract_model->save_end_points(po.get("end_point").c_str()))
    {
        tipl::error() << "failed to save file to " << po.get("end_point") << std::endl;
        return 1;
    }
    if(po.has(("end_point1")) || po.has(("end_point2")))
    {
        std::vector<tipl::vector<3,short> > points1,points2;
        tract_model->to_end_point_voxels(points1,points2);
        ROIRegion end1(handle),end2(handle);
        end1.add_points(std::move(points1));
        end2.add_points(std::move(points2));
        if(po.has(("end_point1")) && !end1.save_region_to_file(po.get("end_point1").c_str()))
        {
            tipl::error() << "failed to save file to " << po.get("end_point1") << std::endl;
            return 1;
        }
        if(po.has(("end_point2")) && !end2.save_region_to_file(po.get("end_point2").c_str()))
        {
            tipl::error() << "failed to save file to " << po.get("end_point2") << std::endl;
            return 1;
        }
    }

    if(po.has("connectivity") && !get_connectivity_matrix(po,handle,tract_file_name,tract_model))
        return 1;
    if(po.has("export") && !export_track_info(po,handle,tract_file_name,tract_model))
        return 1;
    return 0;
}

bool load_roi(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,std::shared_ptr<RoiMgr> roi_mgr)
{
    const int total_count = 20;
    static const char roi_names[total_count][5] = {"roi","roi2","roi3","roi4","roi5","roa","roa2","roa3","roa4","roa5","end","end2","seed","ter","ter2","ter3","ter4","ter5","nend","lim"};
    static const unsigned char type[total_count] = {0,0,0,0,0,1,1,1,1,1,2,2,3,4,4,4,4,4,5,6};

    for(int index = 0;index < total_count;++index)
    if (po.has(roi_names[index]))
    {
        ROIRegion roi(handle);
        QStringList roi_list = QString(po.get(roi_names[index]).c_str()).split("+");
        std::string region_name;
        for(int i = 0;i < roi_list.size();++i)
        {
            ROIRegion other_roi(handle);
            if(!load_region(po,handle,i ? other_roi : roi,roi_list[i].toStdString()))
                return false;
            if(i)
            {
                roi.add_points(std::move(other_roi.region));
                region_name += ",";
            }
            region_name += roi_list[0].toStdString();
        }
        roi_mgr->setRegions(roi.region,roi.dim,roi.to_diffusion_space,type[index],region_name.c_str());
    }
    if(po.has("track_id"))
    {
        tipl::out() << "Consider using action atk for automatic fiber tracking" << std::endl;
        if(!handle->load_track_atlas())
        {
            tipl::error() << handle->error_msg << std::endl;
            return false;
        }
        std::string name = po.get("track_id");
        if(name[0] >= '0' && name[0] <= '9')
        {
            size_t track_id = std::stoi(name);
            if(track_id >= handle->tractography_name_list.size())
            {
                tipl::error() << "invalid track_id";
                return false;
            }
            roi_mgr->tract_name = handle->tractography_name_list[track_id];
        }
        else
            roi_mgr->tract_name = name;
        roi_mgr->use_auto_track = true;
        roi_mgr->tolerance_dis_in_icbm152_mm = po.get("tolerance",16.0f);
    }
    return true;
}

int trk(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle);
int trk(tipl::program_option<tipl::out>& po)
{
    try{
        std::shared_ptr<fib_data> handle = cmd_load_fib(po);
        if(!handle.get())
            return 1;
        return trk(po,handle);
    }
    catch(std::exception const&  ex)
    {
        tipl::error() << "program terminated due to exception: " << ex.what() << std::endl;
    }
    catch(...)
    {
        tipl::error() << "program terminated due to unknown exception" << std::endl;
    }
    return 0;
}

void setup_trk_param(std::shared_ptr<fib_data> handle,ThreadData& tracking_thread,tipl::program_option<tipl::out>& po)
{
    tipl::progress prog("tracking parameters:");
    tracking_thread.param.default_otsu = po.get("otsu_threshold",tracking_thread.param.default_otsu);
    tracking_thread.param.threshold = po.get("fa_threshold",tracking_thread.param.threshold);
    tracking_thread.param.dt_threshold = po.get("dt_threshold",po.has("dt_threshold_index") ? 0.2f : tracking_thread.param.dt_threshold);
    tracking_thread.param.cull_cos_angle = float(std::cos(po.get("turning_angle",0.0)*3.14159265358979323846/180.0));
    tracking_thread.param.step_size = po.get("step_size",tracking_thread.param.step_size);
    tracking_thread.param.smooth_fraction = po.get("smoothing",tracking_thread.param.smooth_fraction);
    tracking_thread.param.min_length = po.get("min_length",handle->min_length());
    tracking_thread.param.max_length = std::max<float>(tracking_thread.param.min_length,po.get("max_length",handle->max_length()));

    tracking_thread.param.random_seed = uint8_t(po.get("random_seed",int(tracking_thread.param.random_seed)));
    tracking_thread.param.tracking_method = uint8_t(po.get("method",int(tracking_thread.param.tracking_method)));
    tracking_thread.param.check_ending = uint8_t(po.get("check_ending",int(0))) && !(po.has("dt_threshold_index"));
    tracking_thread.param.tip_iteration = uint8_t(po.get("tip_iteration", (po.has("track_id") | po.has("dt_metric1") ) ? 4 : 0));

    if (po.has("fiber_count"))
    {
        tracking_thread.param.termination_count = po.get("fiber_count",uint32_t(tracking_thread.param.termination_count));
        tracking_thread.param.stop_by_tract = 1;
        tracking_thread.param.max_seed_count = po.get("seed_count",uint32_t(tracking_thread.param.max_seed_count));
    }
    else
    {
        tracking_thread.param.termination_count = po.get("seed_count",po.has("dt_metric1") ? 10000000 :
                                                        uint32_t(tracking_thread.param.termination_count));
        tracking_thread.param.stop_by_tract = 0;
    }

    if(po.has("parameter_id"))
        tracking_thread.param.set_code(po.get("parameter_id"));
}
extern std::vector<std::string> fa_template_list;
void set_template(std::shared_ptr<fib_data> handle,tipl::program_option<tipl::out>& po)
{
    if(po.has("template"))
    {
        for(size_t id = 0;id < fa_template_list.size();++id)
            tipl::out() << "template " << id << ": " << tipl::split(std::filesystem::path(fa_template_list[id]).filename().string(),'.').front() << std::endl;
        handle->set_template_id(po.get("template",size_t(0)));
    }
}
int trk(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle)
{
    if (po.has("threshold_index"))
    {
        if(!handle->dir.set_tracking_index(po.get("threshold_index")))
        {
            tipl::error() << "cannot find the index" << std::endl;
            return 1;
        }
    }
    if (po.has("dt_metric1") && po.has("dt_metric2"))
    {
        // allow adding other slices for creating new metrics
        {
            if(po.has("subject_demo"))
            {
                auto subject_demo = po.get("subject_demo");
                bool found = false;
                if(std::filesystem::exists(subject_demo))
                {
                    std::ifstream in(subject_demo);
                    std::string line;
                    while(std::getline(in,line))
                    {
                        std::replace(line.begin(),line.end(),',','\t');
                        std::istringstream in2(line);
                        std::string name;
                        in2 >> name;
                        if(po.get("source").find(name) != std::string::npos ||
                           po.get("other_slices").find(name) != std::string::npos)
                        {
                            tipl::out() << "found subject's demographics: " << line << std::endl;
                            found = true;
                            handle->demo = line.substr(name.length()+1);
                        }
                    }
                    if(!found)
                    {
                        tipl::error() << "cannot find subject in " << subject_demo << ". Please make sure that the FIB or NIFTI file name includes subject's id." << std::endl;
                        return 1;
                    }
                }
                else
                    handle->demo = subject_demo;
            }

        }
        auto metric_i = handle->get_name_index(po.get("dt_metric1"));
        auto metric_j = handle->get_name_index(po.get("dt_metric2"));
        if(metric_i == handle->view_item.size() || metric_j == handle->view_item.size())
        {
            tipl::error() << "invalid dt_metric";
            tipl::out() << "available metrics are ";
            for(size_t i = 0;i < handle->view_item.size();++i)
                tipl::out() << handle->view_item[i].name;
            return 1;
        }
        if(handle->view_item[metric_i].name != po.get("dt_metric1"))
            tipl::warning() << "specified " << po.get("dt_metric1") << " but it was not found. The analysis will use " << handle->view_item[metric_i].name;
        if(handle->view_item[metric_j].name != po.get("dt_metric2"))
            tipl::warning() << "specified " << po.get("dt_metric2") << " but it was not found. The analysis will use " << handle->view_item[metric_j].name;
        handle->set_dt_index(std::make_pair(metric_i,metric_j),po.get("dt_threshold_type",0));
    }

    set_template(handle,po);

    ThreadData tracking_thread(handle);
    setup_trk_param(handle,tracking_thread,po);

    {
        tipl::progress prog("setting up regions");
        if(!load_roi(po,handle,tracking_thread.roi_mgr))
            return 1;
    }

    std::shared_ptr<TractModel> tract_model(new TractModel(handle));
    {
        tipl::progress prog("start fiber tracking");
        tracking_thread.run(tipl::max_thread_count = po.get("thread_count",tipl::max_thread_count),true);
        tract_model->report += tracking_thread.report.str();
        if(po.has("report"))
        {
            std::ofstream out(po.get("report").c_str());
            out << tract_model->report;
        }

        tracking_thread.fetchTracks(tract_model.get());
    }


    if(tract_model->get_visible_track_count() && po.has("refine") && (po.get("refine",1) >= 1))
    {
        for(int i = 0;i < po.get("refine",1);++i)
            tract_model->trim();
        tipl::out() << "refine tracking result..." << std::endl;
        tipl::out() << "convert tracks to seed regions" << std::endl;
        tracking_thread.roi_mgr->seeds.clear();
        std::vector<tipl::vector<3,short> > points;
        tract_model->to_voxel(points);
        tract_model->clear();
        tracking_thread.roi_mgr->setRegions(points,3/*seed*/,"refine seeding region");

        tipl::out() << "restart tracking..." << std::endl;
        tracking_thread.run(tipl::max_thread_count,true);
        tracking_thread.fetchTracks(tract_model.get());
        tipl::out() << "finished tracking." << std::endl;

        if(tract_model->get_visible_track_count() == 0)
        {
            tipl::out() << "no tract generated. Terminating..." << std::endl;
            return 0;
        }
    }
    tipl::out() << tract_model->get_visible_track_count() << " tracts are generated using " << tracking_thread.get_total_seed_count() << " seeds."<< std::endl;

    std::string tract_file_name = po.get("source")+".tt.gz";
    bool output_track = true;
    if (po.has("output"))
    {
        std::string output = po.get("output");
        if(output == "no_file")
            output_track = false;
        else
        {
            if(QFileInfo(output.c_str()).isDir())
                tract_file_name = output+"/"+QFileInfo(po.get("source").c_str()).baseName().toStdString() + ".tt.gz";
            else
                tract_file_name = output;
        }
    }

    return trk_post(po,handle,tract_model,tract_file_name,output_track);
}

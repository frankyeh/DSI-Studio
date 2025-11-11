#include <regex>
#include <QFileInfo>
#include <QDir>
#include <QStringList>
#include <iostream>
#include <iterator>
#include <string>
#include "zlib.h"
#include "TIPL/tipl.hpp"
#include "tracking/region/Regions.h"
#include "libs/tracking/tract_model.hpp"
#include "libs/tracking/tracking_thread.hpp"
#include "fib_data.hpp"
#include "atlas.hpp"

#include <filesystem>

#include "SliceModel.h"


bool atl_load_atlas(std::shared_ptr<fib_data> handle,std::string atlas_name,std::vector<std::shared_ptr<atlas> >& atlas_list);
bool load_roi(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,std::shared_ptr<RoiMgr> roi_mgr);

bool load_region(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,
                 ROIRegion& roi,std::string region_text);

void get_regions_statistics(std::shared_ptr<fib_data> handle,const std::vector<std::shared_ptr<ROIRegion> >& regions,
                            std::string& result)
{
    std::vector<std::string> titles;
    std::vector<std::vector<float> > data(regions.size());
    tipl::progress p("for each region");
    for(size_t index = 0;index < regions.size();++index)
    {
        std::vector<std::string> dummy;
        regions[index]->get_quantitative_data(handle,(index == 0) ? titles : dummy,data[index]);
    }
    std::ostringstream out;
    out << "Name";
    for(auto each : regions)
        out << "\t" << each->name;
    out << std::endl;
    for(unsigned int i = 0;i < titles.size();++i)
    {
        out << titles[i];
        for(unsigned int j = 0;j < regions.size();++j)
        {
            out << "\t";
            if(i < data[j].size())
                out << data[j][i];
        }
        out << std::endl;
    }
    result = out.str();
}
void load_nii_label(const char* filename,std::map<int,std::string>& label_map)
{
    std::ifstream in(filename);
    if(in)
    {
        std::string line,txt;
        while(std::getline(in,line))
        {
            if(line.empty() || line[0] == '#')
                continue;
            std::istringstream read_line(line);
            int num = 0;
            read_line >> num >> txt;
            label_map[num] = txt;
        }
    }
}
void load_json_label(const char* filename,std::map<int,std::string>& label_map)
{
    std::ifstream in(filename);
    if(!in)
        return;
    std::string line,label;
    while(std::getline(in,line))
    {
        std::replace(line.begin(),line.end(),'"',' ');
        std::replace(line.begin(),line.end(),'	',' ');
        std::replace(line.begin(),line.end(),',',' ');
        line.erase(std::remove(line.begin(),line.end(),' '),line.end());
        if(line.find("arealabel") != std::string::npos)
        {
            label = line.substr(line.find(":")+1,std::string::npos);
            continue;
        }
        if(line.find("labelIndex") != std::string::npos)
        {
            line = line.substr(line.find(":")+1,std::string::npos);
            std::istringstream in3(line);
            int num = 0;
            in3 >> num;
            label_map[num] = label;
        }
    }
}

std::string get_label_file_name(const std::string& file_name);
void get_roi_label(QString file_name,std::map<int,std::string>& label_map,std::map<int,tipl::rgb>& label_color)
{
    label_map.clear();
    label_color.clear();
    QString base_name = QFileInfo(file_name).completeBaseName();
    if(base_name.endsWith(".nii"))
        base_name.chop(4);
    QString label_file = get_label_file_name(file_name.toStdString()).c_str();
    tipl::out() <<"looking for region label file " << label_file.toStdString() << std::endl;
    if(QFileInfo(label_file).exists())
    {
        load_nii_label(label_file.toStdString().c_str(),label_map);
        tipl::out() <<"label file loaded" << std::endl;
        return;
    }
    label_file = QFileInfo(file_name).absolutePath()+"/"+base_name+".json";
    if(QFileInfo(label_file).exists())
    {
        load_json_label(label_file.toStdString().c_str(),label_map);
        tipl::out() <<"json file loaded " << label_file.toStdString() << std::endl;
        return;
    }
    if(QFileInfo(file_name).fileName().contains("aparc") || QFileInfo(file_name).fileName().contains("aseg")) // FreeSurfer
    {
        tipl::out() <<"using freesurfer labels." << std::endl;
        QFile data(":/data/FreeSurferColorLUT.txt");
        if (data.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            QTextStream in(&data);
            while (!in.atEnd())
            {
                QString line = in.readLine();
                if(line.isEmpty() || line[0] == '#')
                    continue;
                std::istringstream in(line.toStdString());
                int value,r,b,g;
                std::string name;
                in >> value >> name >> r >> g >> b;
                label_map[value] = name;
                label_color[value] = tipl::rgb(uint8_t(r),uint8_t(g),uint8_t(b));
            }
            return;
        }
    }
    tipl::out() <<"no label file found. Use default ROI numbering." << std::endl;
}
bool load_nii(std::shared_ptr<fib_data> handle,
              const std::string& file_name,
              std::vector<SliceModel*>& transform_lookup,
              std::vector<std::shared_ptr<ROIRegion> >& regions,
              std::string& error_msg,
              bool is_mni)
{
    if(QFileInfo(file_name.c_str()).baseName().toLower().contains(".mni."))
    {
        tipl::out() << QFileInfo(file_name.c_str()).baseName().toStdString() <<
                     " has '.mni.' in the file name. It will be treated as mni space image" << std::endl;
        is_mni = true;
    }

    tipl::io::gz_nifti header;
    if (!header.open(file_name,std::ios::in))
    {
        error_msg = header.error_msg;
        return false;
    }
    bool is_4d = header.dim(4) > 1;
    tipl::image<3,unsigned int> from;
    std::string nifti_name = std::filesystem::path(file_name).stem().u8string();
    nifti_name = nifti_name.substr(0,nifti_name.find('.'));

    if(is_4d)
        from.resize(tipl::shape<3>(header.dim(1),header.dim(2),header.dim(3)));
    else
    {
        tipl::image<3> tmp;
        header >> tmp;
        if(std::filesystem::exists(get_label_file_name(file_name)) || tipl::is_label_image(tmp))
            from = tmp;
        else
        {
            tipl::out() << "The NIFTI file does not have a label text file. Using it as a mask.";
            from.resize(tmp.shape());
            for(size_t i = 0;i < from.size();++i)
                from[i] = (tmp[i] > 0.0f ? 1 : 0);
        }
    }

    tipl::vector<3> vs;
    tipl::matrix<4,4> trans_to_mni;
    header >> vs >> trans_to_mni;

    std::vector<unsigned short> value_list;
    std::vector<unsigned short> value_map(std::numeric_limits<unsigned short>::max()+1);

    if(is_4d)
    {
        value_list.resize(header.dim(4));
        for(unsigned int index = 0;index <value_list.size();++index)
        {
            value_list[index] = uint16_t(index);
            value_map[uint16_t(index)] = 0;
        }
    }
    else
    {
        unsigned short max_value = 0;
        for (tipl::pixel_index<3> index(from.shape());index < from.size();++index)
        {
            if(from[index.index()] >= value_map.size())
            {
                error_msg = "exceedingly large value found in the ROI file: ";
                error_msg += std::to_string(from[index.index()]);
                return false;
            }
            value_map[from[index.index()]] = 1;
            max_value = std::max<unsigned short>(uint16_t(from[index.index()]),max_value);
        }
        for(unsigned short value = 1;value <= max_value;++value)
            if(value_map[value])
            {
                value_map[value] = uint16_t(value_list.size());
                value_list.push_back(value);
            }
    }

    bool multiple_roi = value_list.size() > 1;


    tipl::out() << nifti_name << (multiple_roi ? " loaded as multiple ROI file":" loaded as single ROI file") << std::endl;

    std::map<int,std::string> label_map;
    std::map<int,tipl::rgb> label_color;

    std::string des(header.get_descrip());
    if(multiple_roi)
        get_roi_label(file_name.c_str(),label_map,label_color);

    bool need_trans = false;
    tipl::matrix<4,4> to_diffusion_space = tipl::identity_matrix();

    tipl::out() << "FIB file size: " << handle->dim << " vs: " << handle->vs << (handle->is_mni ? " mni space": " not mni space") << std::endl;
    tipl::out() << nifti_name << " size: " << from.shape() << " vs: " << vs << (is_mni ? " mni space": " not mni space (if mni space, add '.mni.' in file name)") << std::endl;

    if(from.shape() != handle->dim)
    {
        tipl::out() << nifti_name << " has a different dimension from the FIB file. need transformation or warping." << std::endl;
        if(handle->is_mni)
        {
            if(!is_mni)
                tipl::out() << "assume " << nifti_name << " is in the mni space (likely wrong. need to check)." << std::endl;
            else
                tipl::out() << nifti_name << " is in the mni space." << std::endl;


            tipl::out() <<"applying " << nifti_name << "'s header srow matrix to align." << std::endl;
            to_diffusion_space = tipl::from_space(trans_to_mni).to(handle->trans_to_mni);
            need_trans = true;
            goto end;
        }
        else
        {
            if(is_mni)
            {
                tipl::out() << "warping " << nifti_name << " from the template space to the native space." << std::endl;
                if(!handle->mni2sub<tipl::interpolation::majority>(from,trans_to_mni))
                {
                    error_msg = handle->error_msg;
                    return false;
                }
                trans_to_mni = handle->trans_to_mni;
                goto end;
            }
            else
            for(unsigned int index = 0;index < transform_lookup.size();++index)
                if(from.shape() == transform_lookup[index]->dim)
                {
                    tipl::out() << "applying previous transformation.";
                    tipl::out() << "tran_to_mni: " << (trans_to_mni = transform_lookup[index]->trans_to_mni);
                    tipl::out() << "to_dif: " << (to_diffusion_space = transform_lookup[index]->to_dif);
                    need_trans = true;
                    goto end;
                }
        }
        error_msg = "No strategy to align ";
        error_msg += nifti_name;
        error_msg += " with FIB. If ";
        error_msg += nifti_name;
        error_msg += " is in the MNI space, ";
        if(tipl::show_prog)
            error_msg += "open it using [Region][Open MNI Region]. If not, insert its reference T1W/T2W using [Slices][Insert T1WT2W] to guide the registration.";
        else
            error_msg += "specify mni in the file name (e.g. region_mni.nii.gz). If not, use --other_slices to load the reference T1W/T2W to guide the registration.";
        return false;
    }
    else
    {
        if(is_mni && !handle->is_mni)
            tipl::out() << "The '.mni.' in the filename is ignored, and " << nifti_name << " is treated as DWI regions because of identical image dimension. " << std::endl;
    }


    end:
    // single region ROI
    if(!multiple_roi)
    {
        regions.push_back(std::make_shared<ROIRegion>(handle));
        regions.back()->name = nifti_name;
        if(need_trans)
        {
            regions.back()->dim = from.shape();
            regions.back()->vs = vs;
            regions.back()->is_diffusion_space = false;
            regions.back()->to_diffusion_space = to_diffusion_space;
            regions.back()->trans_to_mni = trans_to_mni;
        }
        tipl::image<3,unsigned char> mask(from);
        regions.back()->load_region_from_buffer(mask);

        unsigned int color = 0x00FFFFFF;
        unsigned int type = default_id;

        try{
            for(const auto each : tipl::split(std::string(header.get_descrip()),';'))
            {
                auto name_value = tipl::split(each,'=');
                if(name_value.size() != 2)
                    continue;
                if(name_value[0] == "color")
                    std::istringstream(name_value[1]) >> color;
                if(name_value[0] == "roi")
                    std::istringstream(name_value[1]) >> type;
            }
        }catch(...){}
        regions.back()->region_render->color = color;
        regions.back()->regions_feature = uint8_t(type);
        return true;
    }

    std::vector<std::vector<tipl::vector<3,short> > > region_points(value_list.size());
    if(is_4d)
    {
        tipl::progress prog("loading");
        for(size_t region_index = 0;prog(region_index,region_points.size());++region_index)
        {
            header >> from;
            for (tipl::pixel_index<3> index(from.shape());index < from.size();++index)
                if(from[index.index()])
                    region_points[region_index].push_back(index);
        }
    }
    else
    {
        for (tipl::pixel_index<3>index(from.shape());index < from.size();++index)
            if(from[index.index()])
                region_points[value_map[from[index.index()]]].push_back(index);
    }

    for(uint32_t i = 0;i < region_points.size();++i)
        {
            unsigned short value = value_list[i];
            regions.push_back(std::make_shared<ROIRegion>(handle));
            regions.back()->name = (label_map.find(value) == label_map.end() ?
                                    nifti_name + "_" + std::to_string(int(value)): label_map[value]);
            if(need_trans)
            {
                regions.back()->dim = from.shape();
                regions.back()->vs = vs;
                regions.back()->is_diffusion_space = false;
                regions.back()->to_diffusion_space = to_diffusion_space;
                regions.back()->trans_to_mni = trans_to_mni;
            }
            regions.back()->region_render->color = label_color.empty() ? 0x00FFFFFF : label_color[value].color;
            if(!region_points[i].empty())
                regions.back()->add_points(std::move(region_points[i]));
        }
    tipl::out() <<"a total of " << regions.size() << " regions are loaded." << std::endl;
    if(regions.empty())
    {
        error_msg = "empty region file";
        return false;
    }
    return true;
}


extern std::vector<std::shared_ptr<CustomSliceModel> > other_slices;
bool check_other_slices(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle);
bool load_nii(tipl::program_option<tipl::out>& po,
              std::shared_ptr<fib_data> handle,
              const std::string& region_text,
              std::vector<std::shared_ptr<ROIRegion> >& regions)
{
    std::vector<SliceModel*> transform_lookup;
    if(!check_other_slices(po,handle))
        return false;
    for(const auto& each : other_slices)
        transform_lookup.push_back(each.get());

    QStringList str_list = QString(region_text.c_str()).split(",");// splitting actions
    QString file_name = str_list[0];
    std::string error_msg;
    if(!load_nii(handle,file_name.toStdString(),transform_lookup,regions,error_msg,false))
    {
        tipl::error() << error_msg << std::endl;
        return false;
    }

    // now perform actions
    for(int i = 1;i < str_list.size();++i)
    {
        tipl::out() << str_list[i].toStdString() << " applied." << std::endl;
        for(size_t j = 0;j < regions.size();++j)
            regions[j]->perform(str_list[i].toStdString());
    }

    return true;
}


int trk_post(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle,std::shared_ptr<TractModel> tract_model,std::string tract_file_name,bool output_track);
std::shared_ptr<fib_data> cmd_load_fib(tipl::program_option<tipl::out>& po);

bool load_tracts(const char* file_name,std::shared_ptr<fib_data> handle,std::shared_ptr<TractModel> tract_model,std::shared_ptr<RoiMgr> roi_mgr)
{
    if(!std::filesystem::exists(file_name))
    {
        tipl::error() << file_name << " does not exist. terminating..." << std::endl;
        return false;
    }
    if(QFileInfo(file_name).baseName().contains(".mni."))
        tipl::out() << QFileInfo(file_name).baseName().toStdString() <<
                     " has '.mni.' in the file name. It will be treated as mni-space tracts" << std::endl;
    if(!tract_model->load_tracts_from_file(file_name,handle.get(),QFileInfo(file_name).baseName().contains(".mni.")))
    {
        tipl::error() << "cannot read or parse " << file_name << std::endl;
        return false;
    }
    tipl::out() << "A total of " << tract_model->get_visible_track_count() << " tracks loaded" << std::endl;
    if(!roi_mgr->report.empty())
    {
        tipl::out() << "filtering tracts using roi/roa/end regions." << std::endl;
        tract_model->filter_by_roi(roi_mgr);
        tipl::out() << "remaining tract count: " << tract_model->get_visible_track_count() << std::endl;
    }
    return true;
}
int ana_region(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle)
{
    std::vector<std::shared_ptr<ROIRegion> > regions;
    if(po.has("atlas"))
    {
        std::vector<std::shared_ptr<atlas> > atlas_list;
        if(!atl_load_atlas(handle,po.get("atlas"),atlas_list))
            return 1;
        for(unsigned int i = 0;i < atlas_list.size();++i)
        {
            for(unsigned int j = 0;j < atlas_list[i]->get_list().size();++j)
            {
                std::shared_ptr<ROIRegion> region(std::make_shared<ROIRegion>(handle));
                if(!load_region(po,handle,*region.get(),atlas_list[i]->name + ":" + atlas_list[i]->get_list()[j]))
                {
                    tipl::out() << "fail to load the ROI: " << atlas_list[i]->get_list()[j] << std::endl;
                    return 1;
                }
                region->name = atlas_list[i]->get_list()[j];
                regions.push_back(region);
            }

        }
    }
    if(po.has("region"))
    {
        for(const auto& each : tipl::split(po.get("region"),','))
        {
            std::shared_ptr<ROIRegion> region(new ROIRegion(handle));
            if(!load_region(po,handle,*region.get(),each))
            {
                tipl::error() << "fail to load the ROI file." << std::endl;
                return 1;
            }
            region->name = each;
            regions.push_back(region);
        }
    }
    if(po.has("regions"))
    {
        for(const auto& each : tipl::split(po.get("regions"),','))
        {
            if(!load_nii(po,handle,each,regions))
                return 1;
        }
    }
    if(regions.empty())
    {
        tipl::error() << "no region assigned" << std::endl;
        return 1;
    }

    std::string result;
    tipl::out() << "calculating region statistics at a total of " << regions.size() << " regions" << std::endl;
    get_regions_statistics(handle,regions,result);

    std::string file_name(po.get("source"));
    file_name += ".statistics.txt";
    if(po.has("output"))
    {
        std::string output = po.get("output");
        if(QFileInfo(output.c_str()).isDir())
            file_name = output + std::string("/") + std::filesystem::path(file_name).filename().u8string();
        else
            file_name = output;
        if(file_name.find(".txt") == std::string::npos)
            file_name += ".txt";
    }
    tipl::out() << "saving " << file_name << std::endl;
    std::ofstream out(file_name.c_str());
    out << result <<std::endl;
    return 0;
}
void get_tract_statistics(std::shared_ptr<fib_data> handle,
                          const std::vector<std::shared_ptr<TractModel> >& tract_models,
                          std::string& result)
{
    if(tract_models.empty())
        return;
    std::vector<std::vector<std::string> > track_results(tract_models.size());
    {
        tipl::progress p("for each tract");
        for(size_t index = 0;p(index,tract_models.size());++index)
        {
            std::string tmp,line;
            tract_models[index]->get_quantitative_info(handle,tmp);
            std::istringstream in(tmp);
            while(std::getline(in,line))
            {
                if(line.find("\t") == std::string::npos)
                    continue;
                track_results[index].push_back(line);
            }
        }
        if(p.aborted())
            return;
    }
    std::vector<std::string> metrics_name;
    for(unsigned int j = 0;j < track_results[0].size();++j)
        metrics_name.push_back(track_results[0][j].substr(0,track_results[0][j].find("\t")));

    std::ostringstream out;
    out << "Tract Name\t";
    for(unsigned int index = 0;index < tract_models.size();++index)
        out << tract_models[index]->name << "\t";
    out << std::endl;
    for(unsigned int index = 0;index < metrics_name.size();++index)
    {
        out << metrics_name[index];
        for(unsigned int i = 0;i < track_results.size();++i)
            if(index < track_results[i].size())
                out << track_results[i][index].substr(track_results[i][index].find("\t"));
            else
                out << "\t";
        out << std::endl;
    }
    result = out.str();
}
int ana_tract(tipl::program_option<tipl::out>& po,std::shared_ptr<fib_data> handle)
{
    std::shared_ptr<RoiMgr> roi_mgr(new RoiMgr(handle));
    if(!load_roi(po,handle,roi_mgr))
        return 1;


    std::vector<std::string> tract_files;
    if(!po.get_files("tract",tract_files))
    {
        tipl::error() << po.error_msg << std::endl;
        return 1;
    }

    if(tract_files.size() == 0)
    {
        tipl::error() << "no tract file found at --tract" << std::endl;
        return 1;
    }


    std::vector<std::shared_ptr<TractModel> > tracts;
    for(const auto& each : tract_files)
    {
        tracts.push_back(std::make_shared<TractModel>(handle));
        if(!load_tracts(each.c_str(),handle,tracts.back(),roi_mgr))
            return 1;
    }
    tipl::out() << "a total of " << tract_files.size() << " tract file(s) loaded" << std::endl;


    auto tract_cluster = tracts[0]->tract_cluster;
    if(tracts.size() == 1 && !tracts[0]->tract_cluster.empty())
    {
        tipl::out() << "loading cluster information";
        std::vector<std::string> tract_name;
        if(std::filesystem::exists(tract_files[0]+".txt"))
        {
            std::ifstream in(tract_files[0]+".txt");
            tract_name = std::vector<std::string>((std::istream_iterator<std::string>(in)),(std::istream_iterator<std::string>()));
        }
        tracts = TractModel::separate_tracts(tracts[0],tracts[0]->tract_cluster,tract_name);
        if(tracts.size() > 1 && !std::filesystem::exists(tract_files[0]+".txt"))
            tipl::warning() << "cannot find label file: " << tract_files[0] << ".txt";
        tipl::out() << "cluster count: " << tracts.size();

    }

    if(po.has("name"))
    {
        tipl::out() << "open label file: " << po.get("name");
        std::ifstream in(po.get("name"));
        if(!in)
        {
            tipl::error() << "cannot open file:" << po.get("name");
            return 1;
        }
        std::string line;
        for(size_t i = 0;i < tracts.size() && std::getline(in,line);++i)
            tracts[i]->name = line;
    }

    if(po.has("merge_all") && tracts.size() > 1)
    {
        tipl::out() << "merging all tract clusters into a single one";
        for(size_t index = 1;index < tracts.size();++index)
            tracts[0]->add(*tracts[index].get());
        tracts.resize(1);
    }

    if(tracts.size() > 1)
    {
        tipl::out() << "multiple cluster tracts found. only --output, --connectivity, and --export are supported and cannot use other post-tracking routines.";
        tipl::out() << "To use other post-tracking routines, please specify --merge_all to merge all clusters into one single cluster.";

        if(po.has("output"))
        {
            std::string output = po.get("output");
            // accumulate multiple tracts into one probabilistic nifti volume
            if(QString(output.c_str()).endsWith(".nii.gz"))
            {
                tipl::out() << "computing tract probability to " << output << std::endl;
                if(std::filesystem::exists(output))
                    tipl::out() << output << " exists." << std::endl;
                else
                {
                    auto dim = handle->dim;
                    tipl::image<3,uint32_t> accumulate_map(dim);
                    std::mutex add_lock;
                    tipl::adaptive_par_for(tract_files.size(),[&](size_t i)
                    {
                        tipl::out() << "accumulating " << tract_files[i] << "..." <<std::endl;
                        std::vector<tipl::vector<3,short> > points;
                        tracts[i]->to_voxel(points);
                        tipl::image<3,char> tract_mask(dim);
                        for(size_t j = 0;j < points.size();++j)
                        {
                            auto p = points[j];
                            if(dim.is_valid(p))
                                tract_mask[tipl::pixel_index<3>(p[0],p[1],p[2],dim).index()]=1;
                        }
                        std::scoped_lock lock(add_lock);
                        accumulate_map += tract_mask;
                    });
                    tipl::image<3> pdi(accumulate_map);
                    pdi *= 1.0f/float(tract_files.size());
                    if(!(tipl::io::gz_nifti(output,std::ios::out) << handle->bind(pdi)))
                        return 1;
                }
            }
            if(QString(output.c_str()).endsWith(".trk.gz") ||
               QString(output.c_str()).endsWith(".tt.gz"))
            {
                tipl::out() << "saving multiple tracts into one file: " << output;
                if(!TractModel::save_all(output,tracts))
                {
                    tipl::error() << "cannot write to " << output << std::endl;
                    return 1;
                }
            }
        }
        if(po.has("export"))
        {
            std::string result,file_name_stat("stat.txt");
            get_tract_statistics(handle,tracts,result);
            tipl::out() << "saving " << file_name_stat;
            std::ofstream out_stat(file_name_stat.c_str());
            if(!out_stat)
            {
                tipl::out() << "cannot save statistics. please check write permission" << std::endl;
                return false;
            }
            out_stat << result;
        }

        return 0;
    }

    if(po.has("tip_iteration"))
        tracts[0]->trim(po.get("tip_iteration",0));
    return trk_post(po,handle,tracts[0],tract_files[0],false);
}
int exp(tipl::program_option<tipl::out>& po);
int ana(tipl::program_option<tipl::out>& po)
{
    std::shared_ptr<fib_data> handle = cmd_load_fib(po);
    if(!handle.get())
        return 1;
    if(po.has("atlas") || po.has("region") || po.has("regions"))
        return ana_region(po,handle);
    if(po.has("tract"))
        return ana_tract(po,handle);
    if(po.has("info"))
    {
        auto result = evaluate_fib(handle->dim,handle->dir.fa_otsu*0.6f,handle->dir.fa,[handle](size_t pos,unsigned int fib)
                                        {return handle->dir.get_fib(pos,fib);});
        std::ofstream out(po.get("info"));
        out << "fiber coherence index\t" << result << std::endl;
        return 0;
    }
    if(po.has("export"))
        return exp(po);
    tipl::error() << "please specify --tract or --regions" << std::endl;
    return 1;
}

#ifndef BASIC_VOXEL_HPP
#define BASIC_VOXEL_HPP
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/inherit_linearly.hpp>
#include <tipl/tipl.hpp>
#include <string>
#include "tessellated_icosahedron.hpp"
#include "gzip_interface.hpp"
#include "prog_interface_static_link.h"
struct ImageModel;
struct VoxelParam;
class Voxel;
struct VoxelData;
class BaseProcess
{
public:
    BaseProcess(void) {}
    virtual void init(Voxel&) {}
    virtual void run(Voxel&, VoxelData&) {}
    virtual void end(Voxel&,gz_mat_write&) {}
    virtual ~BaseProcess(void) {}
};



struct VoxelData
{
    unsigned int voxel_index;
    std::vector<float> space;
    std::vector<float> odf;
    std::vector<float> fa;
    std::vector<float> rdi;
    std::vector<tipl::vector<3,float> > dir;
    std::vector<short> dir_index;
    float min_odf;
    tipl::matrix<3,3,float> jacobian;

    void init(void)
    {
        std::fill(fa.begin(),fa.end(),0.0);
        std::fill(dir_index.begin(),dir_index.end(),0);
        std::fill(dir.begin(),dir.end(),tipl::vector<3,float>());
    }
};

struct ImageModel;
class Voxel
{
private:
    std::vector<std::shared_ptr<BaseProcess> > process_list;
public:
    tipl::geometry<3> dim;
    tipl::vector<3> vs;
public:

    tipl::image<unsigned char,3> mask;
    void calculate_mask(const tipl::image<float,3>& dwi_sum);
public:
    std::vector<const unsigned short*> dwi_data;
    std::vector<tipl::vector<3,float> > bvectors;
    std::vector<float> bvalues;

    std::string report;
    std::ostringstream recon_report;
    unsigned int thread_count = 1;
    void load_from_src(ImageModel& image_model);
public:
    unsigned char method_id;
    std::vector<float> param;
    tessellated_icosahedron ti;
public:
    std::string file_name;
    bool need_odf = false;
    bool check_btable = true;
    unsigned int max_fiber_number = 5;
    std::vector<std::string> file_list;
public:// DTI
    bool output_diffusivity = false;
    bool output_tensor = false;
public://used in GQI
    bool r2_weighted = false;// used in GQI only
    bool half_sphere = false;
    int b0_index = -1;
    void calculate_sinc_ql(std::vector<float>& sinc_ql);
    void calculate_q_vec_t(std::vector<tipl::vector<3,float> >& q_vector_time);
public://used in GQI
    bool scheme_balance = false;
    bool csf_calibration = false;
public:// odf sharpening
    bool odf_deconvolusion = false;
    bool odf_decomposition = false;
    tipl::vector<3,short> odf_xyz;
public:// gradient deviation
    std::vector<tipl::image<float,3> > new_grad_dev;
    std::vector<tipl::pointer_image<float,3> > grad_dev;
public:// used in QSDR
    float trans_to_mni[16];
    std::string external_template;
    unsigned char reg_method = 0;
    tipl::transformation_matrix<double> qsdr_trans;
    bool output_jacobian = false;
    bool output_mapping = false;
    bool output_rdi = false;
    bool qsdr = false;
    tipl::vector<3,int> csf_pos1,csf_pos2,csf_pos3,csf_pos4;
    double R2;
public: // for QSDR associated T1WT2W
    std::vector<tipl::image<float,3> > other_image;
    std::vector<std::string> other_image_name;
    std::vector<tipl::transformation_matrix<double> > other_image_affine;
public: // for T1W based DMDM
    tipl::image<float,3> t1w,t1wt;
    tipl::vector<3> t1w_vs,t1wt_vs;
    float t1wt_tran[16];
    std::string t1w_file_name;

public: // for fib evaluation
    std::vector<float> fib_fa;
    std::vector<float> fib_dir;
public: // for DDI
    Voxel* study_data = 0;
    std::string study_name;
    bool ddi_type = true; // true:increased or false:decreased
public:
    float z0 = 0.0;
    // other information for second pass processing
    std::vector<float> response_function,free_water_diffusion;
    tipl::image<float,3> qa_map;
    float reponse_function_scaling;
public:// for template creation
    std::vector<std::vector<float> > template_odfs;
    std::string template_file_name;
public:
    std::vector<VoxelData> voxel_data;
public:
    Voxel(void):param(5){}
    template<class ProcessList>
    void CreateProcesses(void)
    {
        process_list.clear();
        boost::mpl::for_each<ProcessList>(boost::ref(*this));
    }

    template<class Process>
    void operator()(Process&)
    {
        process_list.push_back(std::make_shared<Process>());
    }
public:
    void init(void);
    void run(void);
    void end(gz_mat_write& writer);
    BaseProcess* get(unsigned int index);
};

struct terminated_class {
    unsigned int total;
    mutable unsigned int now;
    mutable bool terminated;
    terminated_class(int total_):total(total_),now(0),terminated(false){}
    bool operator!() const
    {
        terminated = prog_aborted();
        return check_prog(std::min(now++,total-1),total);
    }
    ~terminated_class()
    {
        check_prog(total,total);
    }
};



#endif//BASIC_VOXEL_HPP

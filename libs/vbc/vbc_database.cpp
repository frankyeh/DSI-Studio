#include "vbc_database.h"
#include "libs/tracking/tracking_model.hpp"

vbc_database::vbc_database():fib_file(0),num_subjects(0)
{
}
bool vbc_database::load_template(const char* template_name)
{
    fib_file_buffer.reset(new ODFModel);
    if(!fib_file_buffer->fib_data.load_from_file(template_name))
    {
        error_msg = "Invalid template file";
        return false;
    }
    load_template(fib_file_buffer.get());
    return true;
}

bool vbc_database::load_template(ODFModel* fib_file_)
{
    fib_file = fib_file_;
    dim = (fib_file->fib_data.dim);
    num_fiber = fib_file->fib_data.fib.fa.size();
    findex.resize(num_fiber);
    fa.resize(num_fiber);
    for(unsigned int index = 0;index < num_fiber;++index)
    {
        findex[index] = fib_file->fib_data.fib.findex[index];
        fa[index] = fib_file->fib_data.fib.fa[index];
    }
    vi2si.resize(dim.size());
    for(unsigned int index = 0;index < dim.size();++index)
    {
        if(fa[0][index] != 0.0)
        {
            vi2si[index] = si2vi.size();
            si2vi.push_back(index);
        }
    }
    vertices = fib_file->fib_data.fib.odf_table;
    half_odf_size = vertices.size()/2;

    // does it contain subject info?
    MatFile& matfile = fib_file->fib_data.mat_reader;
    subject_qa.clear();
    subject_qa_buffer.clear();
    for(unsigned int index = 0;1;++index)
    {
        std::ostringstream out;
        out << "subject" << index;
        const float* buf = 0;
        unsigned int row,col;
        matfile.get_matrix(out.str().c_str(),row,col,buf);
        if (!buf)
            break;
        subject_qa.push_back(buf);
    }
    num_subjects = subject_qa.size();
    return !subject_qa.empty();
}

void vbc_database::get_data_at(unsigned int index,unsigned int fib,std::vector<float>& data) const
{
    data.clear();
    if(index >= dim.size() || fa[0][index] == 0.0)
        return;
    unsigned int s_index = vi2si[index];
    unsigned int fib_offset = fib*si2vi.size();
    data.reserve(num_subjects);
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        float value = subject_qa[index][s_index+fib_offset];
        if(value != 0.0)
            data.push_back(value);
    }
}
bool vbc_database::is_consistent(MatFile& mat_reader) const
{
    unsigned int row,col;
    const float* odf_buffer;
    mat_reader.get_matrix("odf_vertices",row,col,odf_buffer);
    if (!odf_buffer)
    {
        error_msg = "Invalid subject data format in ";
        return false;
    }
    if(col != vertices.size())
    {
        error_msg = "Inconsistent ODF dimension in ";
        return false;
    }
    for (unsigned int index = 0;index < col;++index,odf_buffer += 3)
    {
        if(vertices[index][0] != odf_buffer[0] ||
           vertices[index][1] != odf_buffer[1] ||
           vertices[index][2] != odf_buffer[2])
        {
            error_msg = "Inconsistent ODF dimension in ";
            return false;
        }
    }
    return true;
}
bool vbc_database::sample_odf(MatFile& mat_reader,std::vector<float>& data)
{
    ODFData subject_odf;
    for(unsigned int index = 0;1;++index)
    {
        std::ostringstream out;
        out << "odf" << index;
        const float* odf = 0;
        unsigned int row,col;
        mat_reader.get_matrix(out.str().c_str(),row,col,odf);
        if (!odf)
            break;
        subject_odf.setODF(index,odf,row*col);
    }
    subject_odf.initializeODF(dim,fa,half_odf_size);
    if(!subject_odf.has_odfs())
    {
        error_msg = "No ODF data in the subject file:";
        return false;
    }

    set_title("load data");
    float max_iso = 0.0;
    for(unsigned int index = 0;index < si2vi.size();++index)
    {
        unsigned int cur_index = si2vi[index];
        const float* odf = subject_odf.get_odf_data(cur_index);
        if(odf == 0)
            continue;
        float min_value = *std::min_element(odf, odf + half_odf_size);
        if(min_value > max_iso)
            max_iso = min_value;
        unsigned int pos = index;
        for(unsigned char fib = 0;fib < num_fiber;++fib,pos += si2vi.size())
        {
            if(fa[fib][cur_index] == 0.0)
                break;
            // 0: subject index 1:findex by s_index (fa > 0)
            data[pos] = odf[findex[fib][cur_index]]-min_value;
        }
    }

    // normalization
    if(max_iso + 1.0 != 1.0)
        image::divide_constant(data.begin(),data.end(),max_iso);
    return true;
}

bool vbc_database::load_subject_files(const std::vector<std::string>& file_names)
{
    if(!fib_file)
        return false;
    unsigned int num_subjects = file_names.size();
    subject_qa.clear();
    subject_qa_buffer.clear();
    subject_qa.resize(num_subjects);
    subject_qa_buffer.resize(num_subjects);
    for(unsigned int index = 0;index < num_subjects;++index)
    {
        subject_qa_buffer[index].resize(num_fiber*si2vi.size());
        subject_qa[index] = &*(subject_qa_buffer[index].begin());
    }
    // load subject data
    for(unsigned int subject_index = 0;check_prog(subject_index,num_subjects);++subject_index)
    {
        MatFile mat_reader;
        if(!mat_reader.load_from_file(file_names[subject_index].c_str()))
        {
            error_msg = "failed to load subject data ";
            error_msg += file_names[subject_index];
            return false;
        }
        // check if the odf table is consistent or not
        if(!is_consistent(mat_reader) ||
           !sample_odf(mat_reader,subject_qa_buffer[subject_index]))
        {
            error_msg += file_names[subject_index];
            return false;
        }
    }
    return true;
}
void vbc_database::save_subject_data(const char* output_name) const
{
    if(!fib_file)
        return;
    // store results
    MatFile& matfile = fib_file->fib_data.mat_reader;
    matfile.write_to_file(output_name);
    for(unsigned int index = 0;check_prog(index,subject_qa.size());++index)
    {
        std::ostringstream out;
        out << "subject" << index;
        matfile.add_matrix(out.str().c_str(),subject_qa[index],num_fiber,si2vi.size());
    }
    begin_prog("output data");
    matfile.close_file();
}

bool vbc_database::single_subject_analysis(const char* file_name)
{
    single_subject.reset(new ODFModel);
    if(!single_subject->load_from_file(file_name))
    {
        error_msg = "fail to load the fib file";
        return false;
    }
    if(!is_consistent(fib_file->fib_data.mat_reader))
    {
        error_msg = "Inconsistent ODF dimension";
        return false;
    }
    std::vector<float> cur_subject_data(num_fiber*si2vi.size());
    if(!sample_odf(fib_file->fib_data.mat_reader,cur_subject_data))
    {
        error_msg += file_name;
        return false;
    }

    std::vector<std::vector<float> > vbc(num_fiber);
    for(int index = 0;index < num_fiber;++index)
    {
        vbc[index].resize(dim.size());
        std::fill(vbc[index].begin(),vbc[index].end(),0.0);
    }
    std::vector<float> population;
    for(unsigned int s_index = 0;s_index < si2vi.size();++s_index)
    {
        unsigned int cur_index = si2vi[s_index];
        for(unsigned int fib = 0,fib_offset = 0;fib < num_fiber;++fib,fib_offset+=si2vi.size())
        {
            population.clear();
            for(unsigned int i = 0;i < subject_qa.size();++i)
            {
                float value = subject_qa[i][s_index];
                population.push_back(value);
            }
        }
    }
    return true;
}

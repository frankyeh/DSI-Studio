#ifndef MAT_FILE_HPP
#define MAT_FILE_HPP
#include <boost/ptr_container/ptr_vector.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept>
#include "gzlib/zlib.h"
#include "prog_interface_static_link.h"

template<typename fun_type>
struct mat_type_info;

template<>
struct mat_type_info<double>
{
    static const unsigned int type = 0;
};
template<>
struct mat_type_info<float>
{
    static const unsigned int type = 10;
};
template<>
struct mat_type_info<unsigned int>
{
    static const unsigned int type = 20;
};

template<>
struct mat_type_info<int>
{
    static const unsigned int type = 20;
};

template<>
struct mat_type_info<short>
{
    static const unsigned int type = 30;
};
template<>
struct mat_type_info<unsigned short>
{
    static const unsigned int type = 40;
};
template<>
struct mat_type_info<unsigned char>
{
    static const unsigned int type = 50;
};
template<>
struct mat_type_info<char>
{
    static const unsigned int type = 50;
};
const unsigned int element_size_array[10] = {8,4,4,2,2,1,0,0,0,0};



class MatMatrix
{
private:
    std::string name;
    unsigned int type;
    unsigned int rows;
    unsigned int cols;
    unsigned int count;
    unsigned int namelen;
    std::vector<char> data_buf;
    char* data_ptr;
private:
    unsigned int get_total_size(unsigned int ty)
    {
        return count*element_size_array[(ty%100)/10];
    }
    template<typename OutType>
    void get_data(OutType out)
    {
        switch (type)
        {
        case 0://double
            std::copy((const double*)data_ptr,((const double*)data_ptr) + count,out);
            break;
        case 10://float
            std::copy((const float*)data_ptr,((const float*)data_ptr) + count,out);
            break;
        case 20://unsigned int
            std::copy((const unsigned int*)data_ptr,((const unsigned int*)data_ptr) + count,out);
            break;
        case 30://short
            std::copy((const short*)data_ptr,((const short*)data_ptr) + count,out);
            break;
        case 40://unsigned short
            std::copy((const unsigned short*)data_ptr,((const unsigned short*)data_ptr) + count,out);
            break;
        case 50://unsigned char
            std::copy((const unsigned char*)data_ptr,((const unsigned char*)data_ptr) + count,out);
            break;
        }
    }
public:
    MatMatrix(void):type(0),rows(0),cols(0),data_ptr(0)
    {}
    MatMatrix(const std::string& name_):namelen(name_.size()+1),name(name_),data_ptr(0) {}

    template<typename Type>
    void assign(const Type* data_ptr_,unsigned int rows_,unsigned int cols_)
    {
        data_ptr = (char*)data_ptr_;
        rows = rows_;
        cols = cols_;
        type = mat_type_info<Type>::type;
    }
    char* get_data(unsigned int get_type)
    {
        if (get_type != 0 && get_type != 10 && get_type != 20 && get_type != 30 && get_type != 40 && get_type != 50)
            return 0;
        // same type or unsigned short v.s. short
        if (get_type == type || (type == 40 && get_type == 30) || (type == 30 && get_type == 40))
            return data_ptr;
        try
        {
            // reallocate memory on if the required space is larger
            bool change_memory = get_total_size(get_type) > get_total_size(type);
            std::vector<char> allocator;
            char* new_data = 0;
            if (change_memory)
            {
                allocator.resize(get_total_size(get_type));
                new_data = &*allocator.begin();
            }
            else
                new_data = &*data_buf.begin();
            switch (get_type)
            {
            case 0://double
                get_data((double*)new_data);
                break;
            case 10://float
                get_data((float*)new_data);
                break;
            case 20://unsigned int
                get_data((unsigned int*)new_data);
                break;
            case 30://short
                get_data((short*)new_data);
                break;
            case 40://unsigned short
                get_data((unsigned short*)new_data);
                break;
            case 50://unsigned char
                get_data((unsigned char*)new_data);
                break;
            }
            if (change_memory)
            {
                std::swap(data_ptr,new_data);
                allocator.swap(data_buf);
            }
            type = get_type;
            return data_ptr;
        }
        catch (...)
        {
            return 0;
        }
    }
    unsigned int get_rows(void) const
    {
        return rows;
    }
    unsigned int get_cols(void) const
    {
        return cols;
    }
    const std::string& get_name(void) const
    {
        return name;
    }
    void write(void* out)
    {
        unsigned int imagf = 0;
        gzwrite(out,(const char*)&type,4);
        gzwrite(out,(const char*)&rows,4);
        gzwrite(out,(const char*)&cols,4);
        gzwrite(out,(const char*)&imagf,4);
        gzwrite(out,(const char*)&namelen,4);
        gzwrite(out,(const char*)&*name.begin(),namelen);
        gzwrite(out,(const char*)data_ptr,rows*cols*element_size_array[(type%100)/10]);
    }
    void fwrite(FILE* out)
    {
        unsigned int imagf = 0;
        ::fwrite((const char*)&type,4,1,out);
        ::fwrite((const char*)&rows,4,1,out);
        ::fwrite((const char*)&cols,4,1,out);
        ::fwrite((const char*)&imagf,4,1,out);
        ::fwrite((const char*)&namelen,4,1,out);
        ::fwrite((const char*)&*name.begin(),1,namelen,out);
        ::fwrite((const char*)data_ptr,element_size_array[(type%100)/10],rows*cols,out);
    }
    bool read(std::istream& in)
    {
        unsigned int imagf = 0;
        in.read((char*)&type,4);
        if (!in || type > 100 || type % 10 > 1)
            return false;
        if (type % 10) // text
            type = 0;
        in.read((char*)&rows,4);
        in.read((char*)&cols,4);
        in.read((char*)&imagf,4);
        in.read((char*)&namelen,4);
        std::vector<char> buffer(namelen+1);
        in.read((char*)&*buffer.begin(),namelen);
        count = rows*cols;
        name = &*buffer.begin();
        set_title(name.c_str());
        if (!in)
            return false;

        try
        {
            std::vector<char> allocator(get_total_size(type));
            allocator.swap(data_buf);
        }
        catch (...)
        {
            return false;
        }
        data_ptr = &*data_buf.begin();
        in.read(data_ptr,get_total_size(type));
        return in;
    }
    bool read(void* in)
    {
        unsigned int imagf = 0;
        if (gzread(in,(char*)&type,4) == -1 || type > 100 || type % 10 > 1)
            return false;
        if (type % 10) // text
            type = 0;
        if (gzread(in,(char*)&rows,4) == -1)
            return false;
        if (gzread(in,(char*)&cols,4) == -1)
            return false;
        if (gzread(in,(char*)&imagf,4) == -1)
            return false;
        if (gzread(in,(char*)&namelen,4) == -1 || namelen > 255)
            return false;
        std::vector<char> buffer(namelen+1);
        if (gzread(in,(char*)&*buffer.begin(),namelen) == -1)
            return false;
        count = rows*cols;
        name = &*buffer.begin();
        set_title(name.c_str());
        try
        {
            std::vector<char> allocator(get_total_size(type));
            allocator.swap(data_buf);
        }
        catch (...)
        {
            return false;
        }
        data_ptr = &*data_buf.begin();
        return gzread(in,(char*)data_ptr,get_total_size(type)) != -1;
    }
    bool read(FILE* in)
    {
        unsigned int imagf = 0;
        if (fread((char*)&type,4,1,in) != 1 || type > 100 || type % 10 > 1)
            return false;
        if (type % 10) // text
            type = 0;
        if (fread((char*)&rows,4,1,in) != 1)
            return false;
        if (fread((char*)&cols,4,1,in) != 1)
            return false;
        if (fread((char*)&imagf,4,1,in) != 1)
            return false;
        if (fread((char*)&namelen,4,1,in) != 1 || namelen > 255)
            return false;
        std::vector<char> buffer(namelen+1);
        if (fread((char*)&*buffer.begin(),1,namelen,in) != namelen)
            return false;
        count = rows*cols;
        name = &*buffer.begin();
        set_title(name.c_str());
        try
        {
            std::vector<char> allocator(get_total_size(type));
            allocator.swap(data_buf);
        }
        catch (...)
        {
            return false;
        }
        data_ptr = &*data_buf.begin();
        return fread((char*)data_ptr,1,get_total_size(type),in) == get_total_size(type);
    }
};


class MatFile
{
private:
    boost::ptr_vector<MatMatrix> dataset;
    std::map<std::string,unsigned int> name_table;
private:
    void* out;
    bool compressed;
public:
    std::string error_msg;
    MatFile(void):out(0){}
    MatFile(const char* file_name):out(0),compressed(false){write_to_file(file_name);}
    bool load_from_file(const char* file_name)
    {
        boost::ptr_vector<MatMatrix> dataset_buf;
        std::string filename = file_name;
        if (filename.length() > 3 &&
                filename[filename.length()-3] == '.' &&
                filename[filename.length()-2] == 'g' &&
                filename[filename.length()-1] == 'z')
        {
            void* in = gzopen(file_name, "rb");
            if (!in)
            {
                error_msg =  "gzopen failed";
                return false;
            }
            for (unsigned int index = 0;!gzeof(in);++index)
            {
                std::auto_ptr<MatMatrix> matrix(new MatMatrix);
                if (!matrix->read(in))
                    break;
                dataset_buf.push_back(matrix.release());
            }
            gzclose(in);
        }
        else
        {
            FILE* in = fopen(filename.c_str(), "rb");
            if (!in)
            {
                error_msg = "fopen failed";
                return false;
            }
            fseek(in,0,SEEK_END);
            unsigned int file_size = ftell(in);
            fseek(in,0,SEEK_SET);
            while (check_prog(ftell(in),file_size))
            {
                std::auto_ptr<MatMatrix> matrix(new MatMatrix);
                if (!matrix->read((FILE*)in))
                    break;
                dataset_buf.push_back(matrix.release());
            }
            fclose(in);
        }
        for (unsigned int index = 0;index < dataset_buf.size();++index)
            name_table[dataset_buf[index].get_name()] = index;
        dataset_buf.swap(dataset);
        return true;
    }
    void close_file(void)
    {
        if(out)
        {
            if (compressed)
                gzclose(out);
            else
                fclose((FILE*)out);
        }
        out = 0;
    }

    bool write_to_file(const char* file_name,bool compressed_ = true)
    {
        compressed = compressed_;
        std::string filename = file_name;
        if (compressed && !
                (filename.length() > 3 &&
                 filename[filename.length()-3] == '.' &&
                 filename[filename.length()-2] == 'g' &&
                 filename[filename.length()-1] == 'z'))
            filename += ".gz";
        close_file();
        if (compressed)
            out = gzopen(filename.c_str(), "wb");
        else
            out = fopen(filename.c_str(), "wb");
        if(out)
            for(int index = 0;index < dataset.size();++index)
                if (compressed)
                    dataset[index].write(out);
                else
                    dataset[index].fwrite((FILE*)out);
        return out;
    }
    ~MatFile()
    {
        close_file();
    }

    template<typename Type>
    void add_matrix(const char* name,const Type* data_ptr,unsigned int rows,unsigned int cols)
    {
        if(!out)
        {
            std::auto_ptr<MatMatrix> matrix(new MatMatrix(name));
            matrix->assign(data_ptr,rows,cols);
            dataset.push_back(matrix.release());
            return;
        }
        MatMatrix matrix(name);
        matrix.assign(data_ptr,rows,cols);
        if (compressed)
            matrix.write(out);
        else
            matrix.fwrite((FILE*)out);
    }

    void* get_matrix(unsigned int index,unsigned int& rows,unsigned int& cols,unsigned int type)
    {
        if (index >= dataset.size())
            return 0;
        rows = dataset[index].get_rows();
        cols = dataset[index].get_cols();
        return dataset[index].get_data(type);
    }
    void* get_matrix(const char* name,unsigned int& rows,unsigned int& cols,unsigned int type)
    {
        std::map<std::string,unsigned int>::const_iterator iter = name_table.find(name);
        if (iter == name_table.end())
            return 0;
        return get_matrix(iter->second,rows,cols,type);
    }
    unsigned int get_matrix_count(void) const
    {
        return dataset.size();
    }
    const char* get_matrix_name(unsigned int index) const
    {
        return dataset[index].get_name().c_str();
    }
    template<typename out_type>
    bool get_matrix(unsigned int index,unsigned int& rows,unsigned int& cols,const out_type*& out)
    {
        out = (const out_type*)get_matrix(index,rows,cols,mat_type_info<out_type>::type);
        return out != 0;
    }
    template<typename out_type>
    bool get_matrix(const char* name,unsigned int& rows,unsigned int& cols,const out_type*& out)
    {
        out = (const out_type*)get_matrix(name,rows,cols,mat_type_info<out_type>::type);
        return out != 0;
    }
    template<typename out_type>
    bool get_matrix(unsigned int index,unsigned int& rows,unsigned int& cols,out_type*& out)
    {
        out = (out_type*)get_matrix(index,rows,cols,mat_type_info<out_type>::type);
        return out != 0;
    }
    template<typename out_type>
    bool get_matrix(const char* name,unsigned int& rows,unsigned int& cols,out_type*& out)
    {
        out = (out_type*)get_matrix(name,rows,cols,mat_type_info<out_type>::type);
        return out != 0;
    }
};






#endif//MAT_FILE_HPP

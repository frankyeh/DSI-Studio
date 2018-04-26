#ifndef GZIP_INTERFACE_HPP
#define GZIP_INTERFACE_HPP
#ifdef WIN32
#include "QtZlib/zlib.h"
#else
#include "zlib.h"
#endif
#include "tipl/tipl.hpp"
#include "prog_interface_static_link.h"
extern bool prog_aborted_;
class gz_istream{
    size_t size_;
    std::ifstream in;
    gzFile handle;
    bool is_gz(const char* file_name)
    {
        std::string filename = file_name;
        if (filename.length() > 3 &&
                filename[filename.length()-3] == '.' &&
                filename[filename.length()-2] == 'g' &&
                filename[filename.length()-1] == 'z')
            return true;
        return false;
    }
public:
    gz_istream(void):size_(0),handle(0){}
    ~gz_istream(void)
    {
        close();
    }

    template<class char_type>
    bool open(const char_type* file_name)
    {
        prog_aborted_ = false;
        in.open(file_name,std::ios::binary);
        unsigned int gz_size = 0;
        if(in)
        {
            in.seekg(-4,std::ios::end);
            size_ = (size_t)in.tellg()+4;
            in.read((char*)&gz_size,4);
            in.seekg(0,std::ios::beg);
        }
        if(is_gz(file_name))
        {
            in.close();
            size_ = gz_size;
            handle = gzopen(file_name, "rb");
            return handle;
        }
        return in.good();
    }
    bool read(void* buf,size_t buf_size)
    {
        check_prog((unsigned int)cur(),(unsigned int)size());
        if(prog_aborted())
            return false;
        if(handle)
        {

            const size_t block_size = 524288000;// 500mb
            while(buf_size > block_size)
            {
                if(gzread(handle,buf,block_size) <= 0)
                {
                    close();
                    return false;
                }
                buf_size -= block_size;
                buf = (char*)buf + block_size;
            }
            if (gzread(handle,buf,(unsigned int)buf_size) <= 0)
            {
                close();
                return false;
            }
            return true;
        }
        else
            if(in)
            {
                in.read((char*)buf,buf_size);
                return in.good();
            }
        return false;
    }
    void seek(long pos)
    {
        if(handle)
        {
            if(gzseek(handle,pos,SEEK_SET) == -1)
                close();
        }
        else
            if(in)
                in.seekg(pos,std::ios::beg);
    }
    void close(void)
    {
        if(handle)
        {
            gzclose(handle);
            handle = 0;
        }
        if(in)
            in.close();
        check_prog(0,0);
    }
    size_t cur(void)
    {
        return handle ? (size_t)gztell(handle):(size_t)in.tellg();
    }
    size_t size(void)
    {
        return size_;
    }

    operator bool() const	{return handle ? true:in.good();}
    bool operator!() const	{return !(handle? true:in.good());}
};

class gz_ostream{
    std::ofstream out;
    gzFile handle;
    bool is_gz(const char* file_name)
    {
        std::string filename = file_name;
        if (filename.length() > 3 &&
                filename[filename.length()-3] == '.' &&
                filename[filename.length()-2] == 'g' &&
                filename[filename.length()-1] == 'z')
            return true;
        return false;
    }
public:
    gz_ostream(void):handle(0){}
    ~gz_ostream(void)
    {
        close();
    }
public:
    template<class char_type>
    bool open(const char_type* file_name)
    {
        if(is_gz(file_name))
        {
            handle = gzopen(file_name, "wb");
            return handle;
        }
        out.open(file_name,std::ios::binary);
        return out.good();
    }
    void write(const void* buf,size_t size)
    {
        if(handle)
        {
            const size_t block_size = 524288000;// 500mb
            while(size > block_size)
            {
                if(gzwrite(handle,buf,block_size) <= 0)
                {
                    close();
                    throw std::runtime_error("Cannot output gz file");
                }
                size -= block_size;
                buf = (const char*)buf + block_size;
            }
            if(gzwrite(handle,buf,(unsigned int)size) <= 0)
                close();
        }
        else
            if(out)
                out.write((const char*)buf,size);
    }
    void close(void)
    {
        if(handle)
        {
            gzclose(handle);
            handle = 0;
        }
        if(out)
            out.close();
    }
    operator bool() const	{return handle? true:out.good();}
    bool operator!() const	{return !(handle? true:out.good());}
};


typedef tipl::io::nifti_base<gz_istream,gz_ostream> gz_nifti;
typedef tipl::io::mat_write_base<gz_ostream> gz_mat_write;
typedef tipl::io::mat_read_base<gz_istream> gz_mat_read;

#endif // GZIP_INTERFACE_HPP

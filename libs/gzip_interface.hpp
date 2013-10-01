#ifndef GZIP_INTERFACE_HPP
#define GZIP_INTERFACE_HPP
#include "gzlib/zlib.h"
#include "image/image.hpp"
#include "prog_interface_static_link.h"

class gz_istream{
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
    gz_istream(void):handle(0){}
    ~gz_istream(void)
    {
        close();
    }

    template<typename char_type>
    bool open(const char_type* file_name)
    {
        if(is_gz(file_name))
        {
            handle = gzopen(file_name, "rb");
            return handle;
        }
        in.open(file_name,std::ios::binary);
        return in;
    }
    void read(void* buf,size_t size)
    {
        char title[] = "reading......";
        title[7+(std::clock()/CLOCKS_PER_SEC)%5] = 0;
        ::set_title(title);
        if(handle)
        {
            if (gzread(handle,buf,size) == -1)
                close();
        }
        else
            if(in)
                in.read((char*)buf,size);
    }
    void seek(size_t pos)
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
    }

    operator bool() const	{return handle ? true:in;}
    bool operator!() const	{return !(handle? true:in);}
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
    template<typename char_type>
    bool open(const char_type* file_name)
    {
        if(is_gz(file_name))
        {
            handle = gzopen(file_name, "wb");
            return handle;
        }
        out.open(file_name,std::ios::binary);
        return out;
    }
    void write(const void* buf,size_t size)
    {
        char title[] = "writing......";
        title[7+(std::clock()/CLOCKS_PER_SEC)%5] = 0;
        ::set_title(title);
        if(handle)
        {
            if(gzwrite(handle,buf,size) == -1)
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
    operator bool() const	{return handle? true:out;}
    bool operator!() const	{return !(handle? true:out);}
};


typedef image::io::nifti_base<gz_istream,gz_ostream> gz_nifti;
typedef image::io::mat_write_base<gz_ostream> gz_mat_write;
typedef image::io::mat_read_base<gz_istream> gz_mat_read;

#endif // GZIP_INTERFACE_HPP

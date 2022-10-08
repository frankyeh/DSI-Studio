#ifndef GZIP_INTERFACE_HPP
#define GZIP_INTERFACE_HPP
#include "zlib.h"
#include "TIPL/tipl.hpp"
#include "prog_interface_static_link.h"
#include <stdio.h>

#define WINSIZE 32768U      /* sliding window size */

struct access_point {
    uint64_t uncompressed_pos = 0;
    uint64_t compressed_pos = 0;
    unsigned char dict32k[WINSIZE];  /* preceding 32K of uncompressed data */
    access_point(void){;}
    access_point(uint64_t up,uint64_t cp,const unsigned char* dict32k_):
        uncompressed_pos(up),compressed_pos(cp)
    {
        std::copy(dict32k_,dict32k_+WINSIZE,dict32k);
    }
};

class inflate_stream{
    z_stream strm;
    bool ok = true;
    std::vector<unsigned char> buf;
public:
    inflate_stream(void);
    inflate_stream(std::shared_ptr<access_point> point);
    ~inflate_stream();
private: // no copy of the class
    inflate_stream(const inflate_stream& rhs);
    void operator=(const inflate_stream& rhs);
public:
    int process(void);
    int process(size_t& cur_uncompressed,size_t& cur_compressed,bool get_access_point);
    void input(const std::vector<unsigned char>& rhs);
    void input(std::vector<unsigned char>&& rhs);
    void output(void* buf,size_t len);
    bool empty(void) const
    {
        return strm.avail_in == 0;
    }
    size_t size_to_extract(void) const
    {
        return strm.avail_out;
    }
    bool at_block_end(void) const
    {
        return strm.data_type == 128;
    }
    void shift_input(size_t shift)
    {
        strm.avail_in -= shift;
        strm.next_in += shift;
    }
};


class gz_istream{
    std::ifstream in;
    std::shared_ptr<inflate_stream> istrm;
    bool is_gz = false;
private:
    size_t file_size = 0;
    size_t cur_input_index = 0;
    size_t cur_uncompressed = 0;
    size_t cur_compressed = 0;
    size_t cur_input_shift = 0; // used when seek
private:
    // read all buffer
    std::shared_ptr<std::thread> readfile_thread;
    bool terminated = false;
    bool reading_buf = false;
    bool read_each_buf(size_t begin_index,size_t n);
private:
    std::vector<std::thread> inflate_threads;
private:
    std::vector<std::vector<unsigned char> > file_buf;
    std::vector<bool> file_buf_ready;
    std::vector<unsigned char> file_buf_ref;
    bool load_file_buf(size_t size);
    bool fetch(void);
private:
    std::map<uint64_t,std::shared_ptr<access_point>,std::greater<uint64_t> > points;
    std::vector<access_point> access;
    void initgz(void);
    void terminate_readfile_thread(void);
    bool jump_to(std::shared_ptr<access_point> p);
public:
    bool sample_access_point = false;
    bool buffer_all = false;
    bool free_on_read = true;
    bool load_index(const char* file_name);
    bool save_index(const char* file_name);
    bool has_access_points(void) const {return !points.empty();}
public:
    ~gz_istream(void){close();}
    bool open(const char* file_name);
    bool read(void* buf_,size_t buf_size);
    bool seek(size_t offset);
    void flush(void);
    void close(void);
    void clear(void){;}
    size_t tell(void) const
    {
        return cur_uncompressed;
    }
    size_t cur_size(void) const
    {
        return cur_compressed;
    }
    size_t size(void) const
    {
        return file_size;
    }
    bool good(void) const
    {
        return (is_gz ? cur_compressed+8 < file_size : in.good());
    }
    bool eof(void) const
    {
        return in.eof();
    }
    operator bool() const	{return good();}
    bool operator!() const	{return !good();}
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
    gz_ostream(void):handle(nullptr){}
    ~gz_ostream(void)
    {
        close();
    }
public:
    bool open(const char* file_name);
    bool write(const void* buf_,size_t size);
    void flush(void);
    void close(void);
    bool good(void) const {return handle ? !gzeof(handle):out.good();}
    operator bool() const	{return good();}
    bool operator!() const	{return !good();}

};


typedef tipl::io::nifti_base<gz_istream,gz_ostream,progress> gz_nifti;
typedef tipl::io::mat_write_base<gz_ostream> gz_mat_write;
typedef tipl::io::mat_read_base<gz_istream,progress> gz_mat_read;

#endif // GZIP_INTERFACE_HPP

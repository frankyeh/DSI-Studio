#include <stdexcept>
#include <chrono>
#include "gzip_interface.hpp"

#define SPAN 8388608L       /* 8MB as the desired distance between access points */


/* microseconds
 * time to inflate a full filebuf=300
 * time to start a thread= 50;   (can be 300)
 * time to read a file buf =30.7709
 * time to reset z_stream (free,iniate, set dictionary)=12
 */



inflate_stream::inflate_stream(void)
{
    strm.zalloc = nullptr;
    strm.zfree = nullptr;
    strm.opaque = nullptr;
    strm.avail_in = 0;
    strm.next_in = nullptr;
    if (inflateInit2(&strm, 47) != Z_OK) //47 detect header
        throw std::runtime_error("inflateInit2 failed");
}

inflate_stream::inflate_stream(std::shared_ptr<access_point> point)
{
    strm.zalloc = nullptr;
    strm.zfree = nullptr;
    strm.opaque = nullptr;
    strm.avail_in = 0;
    strm.next_in = nullptr;
    if (inflateInit2(&strm, -15) != Z_OK)
        throw std::runtime_error("inflateInit2 failed");
    inflateSetDictionary(&strm,point->dict32k, WINSIZE);
}

inflate_stream::~inflate_stream()
{
    inflateEnd(&strm);
}
int inflate_stream::process(void)
{
    return inflate( &strm, Z_NO_FLUSH);
}
int inflate_stream::process(size_t& cur_uncompressed,size_t& cur_compressed,bool get_access_point)
{
    cur_uncompressed += strm.avail_out;
    cur_compressed += strm.avail_in;

    int ret = inflate( &strm, get_access_point ? Z_BLOCK : Z_NO_FLUSH);

    cur_uncompressed -= strm.avail_out;
    cur_compressed -= strm.avail_in;
    return ret;
}
void inflate_stream::input(const std::vector<unsigned char>& rhs)
{
    strm.avail_in = uint32_t(rhs.size());
    strm.next_in = const_cast<unsigned char*>(&rhs[0]);
}

void inflate_stream::input(std::vector<unsigned char>&& rhs)
{
    buf = std::move(rhs);
    input(buf);
}

void inflate_stream::output(void* buf,size_t len)
{
    strm.next_out = reinterpret_cast<unsigned char *>(buf);
    strm.avail_out = uint32_t(len);
}


bool gz_istream::open(const char* file_name)
{
    prog_aborted_ = false;
    in.open(file_name,std::ios::binary);
    if(!in)
        return false;

    // get file size
    {
        in.seekg(0,std::ios::end);
        file_size = size_t(in.tellg());
        in.clear();
        in.seekg(0,std::ios::beg);
    }

    {
        std::string filename = file_name;
        if (filename.length() > 3 &&
            filename[filename.length()-3] == '.' &&
            filename[filename.length()-2] == 'g' &&
            filename[filename.length()-1] == 'z')
            is_gz = true;
    }

    if(!is_gz)
        return in.good();

    file_buf.resize(file_size/size_t(WINSIZE)+1);
    file_buf_ready.resize(file_buf.size());
    initgz();
    return in.good();
}

void gz_istream::initgz(void)
{
    cur_uncompressed = 0;
    cur_compressed = 0;
    cur_input_index = 0;
    istrm.reset(new inflate_stream);
}

void gz_istream::terminate_readfile_thread(void)
{
    if(readfile_thread.get())
    {
        terminated = true;
        readfile_thread->wait();
        readfile_thread.reset();
        terminated = false;
    }
}

bool gz_istream::read_each_buf(size_t begin_index,size_t n)
{
    size_t end_index = std::min<size_t>(file_buf.size(),begin_index+n);

    if(in.tellg() != int64_t(begin_index)*int64_t(WINSIZE))
    {
        in.clear();
        in.seekg(int64_t(begin_index)*int64_t(WINSIZE),std::ios::beg);
    }
    if(!in)
        return false;
    for(; begin_index < end_index && !terminated && !!in; ++begin_index)
        if(!file_buf_ready[begin_index])
        {
            std::vector<unsigned char> buf(WINSIZE);
            in.read(reinterpret_cast<char*>(&buf[0]),WINSIZE);
            if(in.gcount() != WINSIZE)
                buf.resize(size_t(in.gcount()));
            if(buf.empty())
                return false;
            file_buf[begin_index].swap(buf);
            file_buf_ready[begin_index] = true;
        }
        else
        {
            int64_t jump_dis = WINSIZE;
            while(begin_index+1 < end_index && file_buf_ready[begin_index+1])
            {
                jump_dis += WINSIZE;
                ++begin_index;
            }
            in.seekg(jump_dis,std::ios::cur);
        }
    return true;
}

bool gz_istream::load_file_buf(size_t num)
{
    if(reading_buf && cur_input_index) // file reading is lagging
    {
        // wait at most 0.5ms
        for(size_t i = 0;i < 5;++i)
        {
            using namespace std::chrono;
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(0.1ms);
            if(file_buf_ready[cur_input_index])
                return true;
        }
    }
    const size_t num_in_main = 4;

    size_t read_index = cur_input_index;
    size_t num_in_main_thread = std::min(size_t(num_in_main),num); // read at most num_in_main in the main thread
    size_t remain_num = num-num_in_main_thread;          // remaining buf to be read in other thread

    terminate_readfile_thread();

    // read data at main thread
    if(!read_each_buf(read_index,num_in_main_thread))
        return false;
    read_index += num_in_main_thread;
    if(remain_num)
    {
        // read reamining data at another thread
        readfile_thread = std::make_shared<std::future<void> >(std::async(std::launch::async,[&,remain_num,read_index]()
        {
            reading_buf = true;
            read_each_buf(read_index,remain_num);
            reading_buf = false;
        }));
    }
    return true;
}
bool gz_istream::fetch(void)
{
    if(cur_input_index >= file_buf.size())
        return false;

    if(!file_buf_ready[cur_input_index])
    {
        // avail_out: number of bytes to be uncompressed, >>16 = /WINSIZE/2
        size_t num = buffer_all ? file_buf.size() : std::max<size_t>(4,istrm->size_to_extract() >> 16);
        if(!load_file_buf(num))
            return false;
    }

    if(free_on_read)
    {
        if(cur_input_shift)
        {
            // if there is a shift, then the file buffer may be also read by other threads.
            // make a duplicate instead of freeing the space
            auto duplicate = file_buf[cur_input_index];
            istrm->input(std::move(duplicate));
        }
        else
        {
            file_buf_ready[cur_input_index] = false;
            istrm->input(std::move(file_buf[cur_input_index]));
        }
    }
    else
        istrm->input(file_buf[cur_input_index]);

    if(cur_input_shift) // when jumped
    {
        istrm->shift_input(cur_input_shift);
        cur_input_shift = 0;
    }
    ++cur_input_index;
    return true;
}
void gz_istream::flush(void)
{
    for(auto thread: inflate_thread)
        thread->wait();
    inflate_thread.clear();
}

bool gz_istream::read(void* buf,size_t len)
{
    if(!is_gz)
    {
        if(!good() || prog_aborted())
            return false;
        in.read(reinterpret_cast<char*>(buf),uint32_t(len));
        return true;
    }

    size_t max_readsize = WINSIZE << 10; // 32 MB
    while(len > max_readsize)
    {
        if(!reading_buf)
        {
            size_t num = buffer_all ? file_buf.size() : std::max<size_t>(4,len >> 16); // avail_out: number of bytes to be uncompressed, >>16 = /WINSIZE/2
            if(!load_file_buf(num))
                return false;
        }
        if(!read(buf,max_readsize))
            return false;
        len -= max_readsize;
        buf = reinterpret_cast<unsigned char *>(buf) + max_readsize;
    }

    // consider multiple thread reading, at least 64x32K=2MB, has jump points
    if(len > (WINSIZE << 6) && !sample_access_point && !points.empty())
    {
        auto result = points.lower_bound(cur_uncompressed+len);
        if(result != points.end() && result->first > cur_uncompressed)
        {
            auto& point = result->second;
            size_t byte_to_skip = point->uncompressed_pos - cur_uncompressed; // this value is between 0 and len
            size_t next_file_buf_index = point->compressed_pos/WINSIZE;

            // check if all file buffer are ready to be inflated
            bool data_ready = true;
            for(size_t i = cur_input_index;i <= next_file_buf_index;++i)
                if(!file_buf_ready[i])
                {
                    //std::cout << "WAIT FOR DISK READING" << std::endl;
                    data_ready = false;
                    break;
                }
            if(data_ready)
            {
                //std::cout << "MULTITHREAD GZ" << std::endl;
                auto back_upstrm = istrm;
                back_upstrm->output(buf,byte_to_skip);
                size_t index = cur_input_index;

                // start a new thread to inflate data
                inflate_thread.push_back(std::make_shared<std::future<void> >(std::async(std::launch::async,[this,back_upstrm,index] () mutable
                {
                    do{
                        if(back_upstrm->empty())
                        {
                            if(free_on_read)
                            {
                                file_buf_ready[index] = false;
                                back_upstrm->input(std::move(file_buf[index++]));
                            }
                            else
                                back_upstrm->input(file_buf[index++]);
                        }
                    }while(back_upstrm->process() == Z_OK && back_upstrm->size_to_extract());
                    back_upstrm.reset(); // if not reset, the memory will stay with inflate_thread
                })));

                // now we can jump to the next access point and read it from there
                if(!jump_to(point))
                    return false;
                return read(reinterpret_cast<unsigned char *>(buf)+byte_to_skip,len-byte_to_skip);
            }
        }
    }


    if(len == 0)
        return true;

    bool get_access_point = sample_access_point && len > (WINSIZE << 6);
    size_t access_compressed = 0;
    size_t access_uncompressed = 0;
    unsigned char *buf32k = nullptr;

    istrm->output(buf,len);

    do{

        if(istrm->empty() && !fetch())
            return false;

        int ret = istrm->process(cur_uncompressed,cur_compressed,get_access_point);
        check_prog(cur_compressed*100/file_size,100);

        if(ret == Z_STREAM_END)
        {
            flush();
            if(free_on_read)
            {
                file_buf.clear();
                file_buf.resize(file_buf_ready.size());
                file_buf_ready.clear();
                file_buf_ready.resize(file_buf.size());
            }
            break;
        }

        // ret != Z_OK usually due to data corruption
        if(ret != Z_OK || prog_aborted())
            return false;

        if(get_access_point && istrm->at_block_end() && len > istrm->size_to_extract() + WINSIZE)
        {
            access_compressed = cur_compressed;
            access_uncompressed = cur_uncompressed;
            buf32k = reinterpret_cast<unsigned char *>(buf)+len-istrm->size_to_extract()-WINSIZE;
        }

    }while(istrm->size_to_extract());
    if(buf32k)
        points[access_uncompressed] = std::make_shared<access_point>(access_uncompressed,access_compressed,buf32k);
    return true;
}

bool gz_istream::jump_to(std::shared_ptr<access_point> p)
{
    istrm = std::make_shared<inflate_stream>(p);
    cur_input_index = p->compressed_pos/WINSIZE;
    cur_input_shift = p->compressed_pos%WINSIZE;
    cur_uncompressed = p->uncompressed_pos;
    cur_compressed = p->compressed_pos;
    return true;
}

bool gz_istream::seek(size_t offset)
{
    if(offset == cur_uncompressed)
        return true;

    if(!is_gz)
    {
        in.seekg(int64_t(offset),std::ios::beg);
        return !!in;
    }

    auto result = points.lower_bound(offset);
    if(result == points.end())
    {
        if(offset < cur_uncompressed)
        {
            terminate_readfile_thread();
            initgz();
        }
    }
    else
    {
        const auto& point = result->second;
        if(offset < cur_uncompressed ||  // backward seek, no choice but have to jump
           offset - cur_uncompressed > offset - point->uncompressed_pos) // foward seek, see if jumping can lead to a smaller read size
        {
            terminate_readfile_thread();
            if(!jump_to(point))
                return false;
        }
    }
    std::vector<unsigned char> discard(offset-cur_uncompressed);
    return read(&discard[0],discard.size());
}
bool gz_istream::load_index(const char* file_name)
{
    std::ifstream in(file_name,std::ios::binary);
    if(!in)
        return false;
    points.clear();
    while(in)
    {
        std::shared_ptr<access_point> p(new access_point);
        in.read(reinterpret_cast<char*>(&p->compressed_pos),sizeof(uint64_t));
        if(!in)
            break;
        in.read(reinterpret_cast<char*>(&p->uncompressed_pos),sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(p->dict32k),WINSIZE);
        points[p->uncompressed_pos] = p;
    }
    return true;
}
bool gz_istream::save_index(const char* file_name)
{
    std::ofstream out(file_name,std::ios::binary);
    if(!out)
        return false;
    for(auto iter : points)
    {
        out.write(reinterpret_cast<char*>(&iter.second->compressed_pos),sizeof(uint64_t));
        out.write(reinterpret_cast<char*>(&iter.second->uncompressed_pos),sizeof(uint64_t));
        out.write(reinterpret_cast<char*>(iter.second->dict32k),WINSIZE);
    }
    return true;
}
void gz_istream::close(void)
{
    if(is_gz)
    {
        flush();
        terminate_readfile_thread();
    }
    check_prog(0,0);
}



bool gz_ostream::open(const char* file_name)
{
    if(is_gz(file_name))
    {
        handle = gzopen(file_name, "wb");
        return handle;
    }
    out.open(file_name,std::ios::binary);
    return out.good();
}
void gz_ostream::write(const void* buf_,size_t size)
{
    const char* buf = reinterpret_cast<const char*>(buf_);
    if(handle)
    {
        const size_t block_size = 104857600;// 500mb
        while(size > block_size)
        {
            if(gzwrite(handle,buf,block_size) <= 0)
            {
                close();
                throw std::runtime_error("Cannot output gz file");
            }
            size -= block_size;
            buf = buf + block_size;
        }
        if(gzwrite(handle,buf,uint32_t(size)) <= 0)
            close();
    }
    else
        if(out)
            out.write(buf,uint32_t(size));
}
void gz_ostream::flush(void)
{
    if(handle)
        gzflush(handle,Z_FULL_FLUSH);
    else
    if(out)
        out.flush();
}
void gz_ostream::close(void)
{
    if(handle)
    {
        gzclose(handle);
        handle = nullptr;
    }
    if(out)
        out.close();
}

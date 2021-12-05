#ifndef PROG_INTERFACE_STATIC_LINKH
#define PROG_INTERFACE_STATIC_LINKH
#include <tipl/tipl.hpp>
#include <string>
extern bool has_gui;
void begin_prog(const char* title = nullptr,bool always_show_dialog = false);
void set_title(const char* title);
bool check_prog(unsigned int now,unsigned int total);
template<typename value_type1,typename value_type2>
bool check_prog(value_type1 now,value_type2 total)
{
    return check_prog(uint32_t(now),uint32_t(total));
}

void close_prog();
bool prog_aborted(void);
bool is_running(void);
template<typename fun_type,typename terminated_class>
bool run_prog(const char* msg,fun_type fun,terminated_class& terminated)
{
    if(!has_gui)
    {
        std::cout << msg << std::endl;
        fun();
        return true;
    }
    begin_prog(msg);
    bool ended = false;
    tipl::par_for(2,[&](int i)
    {
        if(i)
        {
            fun();
            ended = true;
        }
        else
        {
            while(!ended)
            {
                check_prog(1,2);
                if(prog_aborted())
                {
                    terminated = true;
                    ended = true;
                }
            }
        }
    });
    close_prog();
    return !prog_aborted();
}

struct prog_init{
    prog_init(const char* status)
    {
        begin_prog(status);
    }
    prog_init(const char* status,bool always_show_dialog)
    {
        begin_prog(status,always_show_dialog);
    }
    prog_init(const char* status1,const char* status2)
    {
        std::string s(status1);
        s += status2;
        begin_prog(s.c_str());
    }
    ~prog_init(void)
    {
        check_prog(0,0);
    }
};

struct unique_prog{
    unique_prog(const char* status)
    {
        begin_prog(status);
        has_gui = false;
    }
    template<typename value_type1,typename value_type2>
    bool operator()(value_type1 now,value_type2 total)
    {
        has_gui = true;
        bool ret = check_prog(uint32_t(now),uint32_t(total));
        has_gui = false;
        return ret;
    }
    ~unique_prog(void)
    {
        has_gui = true;
        check_prog(0,0);
    }
};


#endif


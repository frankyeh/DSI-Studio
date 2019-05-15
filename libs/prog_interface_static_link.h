#ifndef PROG_INTERFACE_STATIC_LINKH
#define PROG_INTERFACE_STATIC_LINKH
#include <tipl/tipl.hpp>
void begin_prog(const char* title = 0,bool lock = false);
void set_title(const char* title);
bool check_prog(unsigned int now,unsigned int total);
void close_prog();
bool prog_aborted(void);
bool is_running(void);

template<typename fun_type,typename terminated_class>
bool run_prog(const char* msg,fun_type fun,terminated_class& terminated)
{
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
    return !prog_aborted();
}
#endif


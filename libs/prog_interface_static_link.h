#ifndef PROG_INTERFACE_STATIC_LINKH
#define PROG_INTERFACE_STATIC_LINKH
#include <tipl/tipl.hpp>
#include <string>
extern bool has_gui;

bool is_main_thread(void);

class progress{
private:
    static void update_prog(bool show_now = false);
    static void begin_prog(bool show_now = false);
    static std::string get_status(void);
    static bool check_prog(unsigned int now,unsigned int total);
    static std::vector<std::string> status_list;
public:
    progress(const char* status,bool show_now = false)
    {
        std::cout << status << std::endl;
        status_list.push_back(status);
        begin_prog(show_now);
    }
    progress(const char* status1,const char* status2)
    {
        std::string s(status1);
        s += status2;
        std::cout << s << std::endl;
        status_list.push_back(s);
        begin_prog();
    }
    static void show(const char* status,bool show_now = false);
    static void show(const std::string& str,bool show_now = false){return show(str.c_str(),show_now);}
    static bool running(void) {return !status_list.empty();}
    static bool aborted(void);
    template<typename value_type1,typename value_type2>
    static bool at(value_type1 now,value_type2 total)
    {
        return check_prog(uint32_t(now),uint32_t(total));
    }
    ~progress(void);
public:
    template<typename fun_type,typename terminated_class>
    static bool run(const char* msg,fun_type fun,terminated_class& terminated)
    {
        if(!has_gui)
        {
            std::cout << msg << std::endl;
            fun();
            return true;
        }
        progress prog_(msg);
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
                    progress::at(1,2);
                    if(progress::aborted())
                    {
                        terminated = true;
                        ended = true;
                    }
                }
            }
        });
        return !progress::aborted();
    }
};





#endif


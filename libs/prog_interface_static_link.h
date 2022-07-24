#ifndef PROG_INTERFACE_STATIC_LINKH
#define PROG_INTERFACE_STATIC_LINKH
#include "TIPL/tipl.hpp"
#include <string>
extern bool has_gui;

bool is_main_thread(void);

class progress{
private:
    static void update_prog(bool show_now = false);
    static void begin_prog(const char* status,bool show_now = false);
    static std::string get_status(void);
    static bool check_prog(unsigned int now,unsigned int total);
    static std::vector<std::string> status_list,at_list;
    static void print_status(const char* status,bool node = true)
    {
        std::istringstream in(status);
        std::string line;
        while(std::getline(in,line))
        {
            for(size_t i = 0;i < status_list.size();++i)
                std::cout << "│";
            if(node)
                std::cout << "┌";
            std::cout << line << std::endl;
            node = false;
        }
    }
public:
    progress(void){}
    progress(const char* status,bool show_now = false)
    {
        print_status(status);
        begin_prog(status,show_now);
    }
    progress(const char* status1,const char* status2,bool show_now = false)
    {
        std::string s(status1);
        s += status2;
        print_status(s.c_str());
        begin_prog(s.c_str(),show_now);
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
    template<typename value_type1,typename value_type2>
    bool operator()(value_type1 now,value_type2 total) const
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
            if(!i)
            {
                fun();
                ended = true;
            }
            else
            {
                size_t i = 0;
                while(!ended)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    progress::at(i,i+1);
                    if(progress::aborted())
                    {
                        terminated = true;
                        ended = true;
                    }
                    ++i;
                }
            }
        });
        return !progress::aborted();
    }
};

class show_progress{
    std::ostringstream s;
public:
    show_progress& operator<<(std::ostream& (*var)(std::ostream&))
    {
        progress::show(s.str().c_str());
        return *this;
    }
    template<typename type>
    show_progress& operator<<(const type& v)
    {
        s << v;
        return *this;
    }
};

#endif


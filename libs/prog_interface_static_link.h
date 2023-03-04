#ifndef PROG_INTERFACE_STATIC_LINKH
#define PROG_INTERFACE_STATIC_LINKH
#include "TIPL/tipl.hpp"
#include <string>
extern bool has_gui;

class progress{
    bool prog_aborted_ = false;
private:
    static void update_prog(bool show_now = false);
    static void begin_prog(const char* status,bool show_now = false);
    static std::string get_status(void);
    static bool check_prog(unsigned int now,unsigned int total);
    static std::vector<std::string> status_list,at_list;
public:
    static void print(const char* status,bool head_node, bool tail_node)
    {
        std::istringstream in(status);
        std::string line;
        while(std::getline(in,line))
        {
            if(line.empty())
                continue;
            std::string head;
            for(size_t i = 0;i < status_list.size();++i)
                head += "| ";
            if(head_node)
                head += "+ ";
            if(tail_node)
                head += "|_";
            if(!has_gui) // enable color output in command line
            {
                if(head_node)
                {
                    head += "\033[1;34m"; // blue
                    line += "\033[0m";
                }
                if(line[0] == 'E' || line[0] == 'W' ) // Error
                {
                    head += "\033[1;31m"; // red
                    line += "\033[0m";
                }
            }
            if(!tipl::is_main_thread<0>())
                head += "[thread]";
            std::cout << head + line << std::endl;
            head_node = false;
        }
    }
public:
    progress(void){}
    progress(const char* status,bool show_now = false)
    {
        print(status,true,false);
        begin_prog(status,show_now);
    }
    progress(const char* status1,const char* status2,bool show_now = false)
    {
        std::string s(status1);
        s += status2;
        print(s.c_str(),true,false);
        begin_prog(s.c_str(),show_now);
    }
    static bool is_running(void) {return !status_list.empty();}
    bool aborted(void);
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
        progress prog(msg);
        if(!has_gui)
        {
            fun();
            return true;
        }
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
                    std::this_thread::yield();
                    prog(i,i+1);
                    if(prog.aborted())
                    {
                        terminated = true;
                        ended = true;
                    }
                    ++i;
                }
            }
        });
        return !prog.aborted();
    }
};

class show_progress{
    std::ostringstream s;
    public:
        ~show_progress()
        {
            auto str = s.str();
            if(str.empty())
                return;
            if(str.back() == '\n')
                str.pop_back();
            progress::print(str.c_str(),false,false);
        }
        show_progress& operator<<(std::ostream& (*var)(std::ostream&))
        {
            s << var;
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


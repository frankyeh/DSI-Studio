#ifndef PROGRAM_OPTION_HPP
#define PROGRAM_OPTION_HPP
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <set>
#include "prog_interface_static_link.h"
class program_option{
    std::vector<std::string> names;
    std::vector<std::string> values;
    std::vector<char> used;
    std::set<std::string> not_found_names;
    std::string no_value;
    bool add_option(const std::string& str)
    {
        if(str.length() < 3 || str[0] != '-' || str[1] != '-')
            return false;
        auto pos = std::find(str.begin(),str.end(),'=');
        if(pos == str.end())
            return false;
        names.push_back(std::string(str.begin()+2,pos));
        values.push_back(std::string(pos+1,str.end()));
        used.push_back(0);
        return true;
    }
public:
    struct program_option_assign{
        const char* name = nullptr;
        program_option* po = nullptr;
        program_option_assign(const char* name_,program_option* po_):name(name_),po(po_){}
        template<typename T> void operator=(const T& rhs)
        {
            std::ostringstream out;
            out << rhs;
            po->set(name,out.str());
        }
    };
    inline program_option_assign operator[](const char* name)     {return program_option_assign(name,this);}
public:
    std::string error_msg;

    ~program_option(void)
    {
        for(size_t i = 0;i < used.size();++i)
            if(!used[i])
            {
                const std::string& str1 = names[i];
                std::map<int,std::string,std::greater<int> > candidate_list;
                for(const auto& str2 : not_found_names)
                {
                    int c = -std::abs(int(str1.length())-int(str2.length()));
                    size_t common_length = std::min(str1.length(),str2.length());
                    for(size_t j = 0;j < common_length;++j)
                    {
                        if(str1[j] == str2[j])
                            ++c;
                        if(str1[str1.length()-1-j] == str2[str2.length()-1-j])
                            ++c;
                    }
                    candidate_list[c] = str2;
                }
                std::string prompt_msg;
                if(!candidate_list.empty() && candidate_list.begin()->first > 0)
                {
                    prompt_msg = "Did you mean --";
                    prompt_msg += candidate_list.begin()->second;
                    prompt_msg += " ?";
                }
                show_progress() << "Warning: --" << str1 << " is not used/recognized. " << prompt_msg << std::endl;
            }
    }
    void clear(void)
    {
        names.clear();
        values.clear();
        used.clear();
    }

    bool parse(int ac, char *av[])
    {
        clear();
        if(ac == 2) // command from log file
        {
            std::ifstream in(av[1]);
            std::string line;
            while(std::getline(in,line))
            {
                line = std::string("--")+line;
                add_option(line);
            }
        }
        else
        for(int i = 1;i < ac;++i)
        {
            std::string str(av[i]);
            if(!add_option(str))
            {
                error_msg = "cannot parse: ";
                error_msg += str;
                return false;
            }
        }
        return true;
    }
    bool parse(const std::string& av)
    {
        clear();
        std::istringstream in(av);
        while(in)
        {
            std::string str;
            in >> str;
            if(str.find('"') != std::string::npos)
            {
                str.erase(str.find('"'),1);
                while(in)
                {
                    std::string other_str;
                    in >> other_str;
                    str += " ";
                    str += other_str;
                    if(other_str.find('"') != std::string::npos)
                    {
                        str.erase(str.find('"'),1);
                        break;
                    }
                }
            }
            if(!str.empty() && !add_option(str))
            {
                error_msg = "cannot parse: ";
                error_msg += str;
                return false;
            }
        }
        return true;
    }

    bool check(const char* name)
    {
        if(!has(name))
        {
            show_progress() << "please specify --" << name << std::endl;
            return false;
        }
        return true;
    }

    bool has(const char* name)
    {
        std::string str_name(name);
        for(size_t i = 0;i < names.size();++i)
            if(names[i] == str_name)
                return true;
        not_found_names.insert(name);
        return false;
    }

    void get_wildcard_list(std::vector<std::pair<std::string,std::string> >& wlist) const
    {
        for(size_t i = 0;i < names.size();++i)
            if(values[i].find('*') != std::string::npos)
                wlist.push_back(std::make_pair(names[i],values[i]));
    }

    void set_used(char value)
    {
        std::fill(used.begin(),used.end(),value);
    }
    void set(const char* name,const std::string& value)
    {
        std::string str_name(name);
        for(size_t i = 0;i < names.size();++i)
            if(names[i] == str_name)
            {
                values[i] = value;
                used[i] = 0;
                return;
            }
        names.push_back(name);
        values.push_back(value);
        used.push_back(0);
    }

    const std::string& get(const char* name)
    {
        std::string str_name(name);
        for(size_t i = 0;i < names.size();++i)
            if(names[i] == str_name)
            {
                if(!used[i])
                {
                    used[i] = 1;
                    show_progress() << name << "=" << values[i] << std::endl;
                }
                return values[i];
            }
        not_found_names.insert(name);
        return no_value;
    }

    std::string get(const char* name,const char* df_ptr)
    {
        std::string str_name(name);
        std::string df_value(df_ptr);
        for(size_t i = 0;i < names.size();++i)
            if(names[i] == str_name)
            {
                if(!used[i])
                {
                    used[i] = 1;
                    show_progress() << name << "=" << values[i] << std::endl;
                }
                return values[i];
            }
        not_found_names.insert(name);
        show_progress() << name << "=" << df_value << std::endl;
        return df_value;
    }

    template<typename value_type>
    value_type get(const char* name,value_type df)
    {
        std::string str_name(name);
        for(size_t i = 0;i < names.size();++i)
            if(names[i] == str_name)
            {
                if(!used[i])
                {
                    used[i] = 1;
                    show_progress() << name << "=" << values[i] << std::endl;
                }
                std::istringstream(values[i]) >> df;
                return df;
            }
        not_found_names.insert(name);
        show_progress() << name << "=" << df << std::endl;
        return df;
    }
};
#endif // PROGRAM_OPTION_HPP


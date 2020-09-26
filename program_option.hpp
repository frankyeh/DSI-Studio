#ifndef PROGRAM_OPTION_HPP
#define PROGRAM_OPTION_HPP
#include <sstream>
#include <vector>
#include <iostream>

class program_option{
    std::vector<std::string> names;
    std::vector<std::string> values;
    std::vector<char> used;
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
    std::string error_msg;

    ~program_option(void)
    {
        for(int i = 0;i < used.size();++i)
            if(!used[i])
            {
                std::cout << "Warning: --" << names[i] << " is not used. Please check command line syntax." << std::endl;
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
            if(!str.empty() && !add_option(str))
            {
                error_msg = "cannot parse: ";
                error_msg += str;
                return false;
            }
        }
        return true;
    }

    bool has(const char* name)
    {
        std::string str_name(name);
        for(int i = 0;i < names.size();++i)
            if(names[i] == str_name)
                return true;
        return false;
    }

    void set(const char* name,const std::string& value)
    {
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
                    std::cout << name << "=" << values[i] << std::endl;
                }
                return values[i];
            }
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
                    std::cout << name << "=" << values[i] << std::endl;
                }
                return values[i];
            }
        std::cout << name << "=" << df_value << std::endl;
        return df_value;
    }

    template<class value_type>
    value_type get(const char* name,value_type df)
    {
        std::string str_name(name);
        for(size_t i = 0;i < names.size();++i)
            if(names[i] == str_name)
            {
                if(!used[i])
                {
                    used[i] = 1;
                    std::cout << name << "=" << values[i] << std::endl;
                }
                std::istringstream(values[i]) >> df;
                return df;
            }
        std::cout << name << "=" << df << std::endl;
        return df;
    }
};


extern program_option po;
#endif // PROGRAM_OPTION_HPP


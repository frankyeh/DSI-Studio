#ifndef PROGRAM_OPTION_HPP
#define PROGRAM_OPTION_HPP
#include <map>
#include <sstream>
class program_option{
    std::map<std::string,std::string> options;
public:
    void init(int ac, char *av[])
    {
        options.clear();
        for(int i = 1;i < ac;++i)
        {
            std::string str(av[i]);
            if(str.length() < 3 || str[0] != '-' || str[1] != '-')
                continue;
            auto pos = std::find(str.begin(),str.end(),'=');
            if(pos == str.end())
                continue;
            options[std::string(str.begin()+2,pos)] = std::string(pos+1,str.end());
        }
    }
    bool has(const char* name)
    {
        return options.find(name) != options.end();
    }

    std::string get(const char* name)
    {
        std::string df;
        auto value = options.find(name);
        if(value != options.end())
            df = value->second;
        return df;
    }

    std::string get(const char* name,const char* df_ptr)
    {
        std::string df;
        auto value = options.find(name);
        if(value != options.end())
            df = value->second;
        else
            df = df_ptr;
        return df;
    }
    template<typename value_type>
    value_type get(const char* name,value_type df)
    {
        auto value = options.find(name);
        if(value != options.end())
            std::istringstream(value->second) >> df;
        return df;
    }
};


extern program_option po;
#endif // PROGRAM_OPTION_HPP


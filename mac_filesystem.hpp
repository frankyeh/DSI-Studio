#ifndef MAC_FILESYSTEM_HPP
#define MAC_FILESYSTEM_HPP
#include <QFileInfo>
#include <string>
#ifndef __APPLE__

#include <filesystem>

#else

namespace std{


    namespace filesystem{

        bool exists(const std::string& file_name)
        {
            return QFileInfo(file_name.c_str()).exists();
        }
        size_t file_size(const std::string& file_name)
        {
            return size_t(QFileInfo(file_name.c_str()).size());
        }

        bool remove(const std::string& file_name)
        {
            return QFile::remove(file_name.c_str());
        }

        struct path_warpper{
            std::string name;
            path_warpper(const std::string& file_name_):name(file_name_){}

            path_warpper filename(void)
            {
                return path_warpper(QFileInfo(name.c_str()).fileName().toStdString());
            }
            const std::string& string(void) const
            {
                return name;
            }

        };

        path_warpper path(const std::string& file_name)
        {
            return path_warpper(file_name);
        }

    }
}

#endif



#endif // MAC_FILESYSTEM_HPP

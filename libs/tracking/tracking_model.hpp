#ifndef ODF_MODEL_HPP
#define ODF_MODEL_HPP
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/noncopyable.hpp>
#include "image/image.hpp"
#include "roi.hpp"
#include "tracking_method.hpp"
#include "fib_data.hpp"
#include "libs/prog_interface_static_link.h"
#include "libs/vbc/vbc_database.h"

template<typename input_iterator,typename output_iterator>
void gradient(input_iterator from,input_iterator to,output_iterator out)
{
	if(from == to)
		return;
	--to;
	if(from == to)
		return;
	*out = *(from+1);
	*out -= *(from);
	output_iterator last = out + (to-from);
	*last = *to;
	*last -= *(to-1);
	input_iterator pre_from = from;
	++from;
	++out;
	input_iterator next_from = from;
	++next_from;
	for(;from != to;++out)
	{
		*out = *(next_from);
		*out -= *(pre_from);
		*out /= 2.0;
		pre_from = from;
		from = next_from;
		++next_from;
	}
}


class ODFModel : public boost::noncopyable
{
public:
    FibData fib_data;
    std::auto_ptr<vbc_database> vbc;
public:
    bool has_vbc(void) const{return vbc.get();}
public:

    void get_tract_data(const std::vector<float>& tract,
                        unsigned int index_num,
                        std::vector<float>& data)
    {
        data.clear();
        if(index_num >= fib_data.view_item.size())
            return;
        data.resize(tract.size()/3);
        for (unsigned int data_index = 0,index = 0;index < tract.size();index += 3,++data_index)
            image::linear_estimate(fib_data.view_item[index_num].image_data,&tract[index],data[data_index]);
    }

    unsigned int get_name_index(const std::string& index_name) const
    {
        for(unsigned int index_num = 0;index_num < fib_data.view_item.size();++index_num)
            if(fib_data.view_item[index_num].name == index_name)
                return index_num;
        return fib_data.view_item.size();
    }

    void get_tracts_data(
            const std::vector<std::vector<float> >& tracts,
            const std::string& index_name,
            std::vector<std::vector<float> >& data)
    {
        data.clear();
        unsigned int index_num = get_name_index(index_name);
        if(index_num == fib_data.view_item.size())
            return;

        data.resize(tracts.size());
        for (unsigned int i = 0;i <tracts.size();++i)
            get_tract_data(tracts[i],index_num,data[i]);
    }

    void get_tract_fa(const std::vector<float>& tract,
                      float threshold,float cull_angle,
                      std::vector<float>& data)
    {
        unsigned int count = tract.size()/3;
        data.resize(count);
        if(tract.empty())
            return;
        std::vector<image::vector<3,float> > gradient(count);
        const float (*tract_ptr)[3] = (const float (*)[3])&tract[0];
        ::gradient(tract_ptr,tract_ptr+count,gradient.begin());

        float prev_info = threshold;
        for (unsigned int point_index = 0,tract_index = 0;
             point_index < count;++point_index,tract_index += 3)
        {
            image::interpolation<image::linear_weighting,3> tri_interpo;
            gradient[point_index].normalize();
            if (tri_interpo.get_location(fib_data.dim,&tract[tract_index]))
            {
                float value,average_value = 0.0;
                float sum_value = 0.0;
                for (unsigned int index = 0;index < 8;++index)
                {
                    if ((value = fib_data.get_directional_fa(tri_interpo.dindex[index],gradient[point_index],threshold,cull_angle)) == 0.0)
                        continue;
                    average_value += value*tri_interpo.ratio[index];
                    sum_value += tri_interpo.ratio[index];
                }
                if (sum_value > 0.5)
                    data[point_index] = prev_info = average_value/sum_value;
				else
                data[point_index] = threshold;
            }
            else
                data[point_index] = threshold;
        }
    }
    void get_tracts_fa(const std::vector<std::vector<float> >& tracts,
                      float threshold,float cull_angle,
                      std::vector<std::vector<float> >& data)
    {
        data.resize(tracts.size());
        for(unsigned int index = 0;index < tracts.size();++index)
            get_tract_fa(tracts[index],threshold,cull_angle,data[index]);
    }

    double get_spin_volume(
            const std::vector<std::vector<float> >& tracts,
            float threshold,float cull_angle)
    {

        std::map<image::vector<3,short>,image::vector<3,float> > passing_regions;
        for (unsigned int i = 0;i < tracts.size();++i)
        {
            std::vector<image::vector<3,float> > point(tracts[i].size() / 3);
            std::vector<image::vector<3,float> > gradient(tracts[i].size() / 3);
            for (unsigned int j = 0,index = 0;j < tracts[i].size();j += 3,++index)
                point[index] = &(tracts[i][j]);

            ::gradient(point.begin(),point.end(),gradient.begin());

            for (unsigned int j = 0;j < point.size();++j)
            {
                gradient[j].normalize();
                point[j] += 0.5;
                point[j].floor();
                passing_regions[image::vector<3,short>(point[j])] += gradient[j];
            }
        }
        double result = 0.0;
        std::map<image::vector<3,short>,image::vector<3,float> >::iterator iter = passing_regions.begin();
        std::map<image::vector<3,short>,image::vector<3,float> >::iterator end = passing_regions.end();
        for (;iter != end;++iter)
        {
            iter->second.normalize();
            result += fib_data.get_directional_fa(
                          image::pixel_index<3>(iter->first[0],iter->first[1],iter->first[2],fib_data.dim).index(),
                          iter->second,threshold,cull_angle);
        }
        return result;
    }
    void getSlicesDirColor(unsigned short order,unsigned int* pixels) const
    {
        for (unsigned int index = 0;index < fib_data.total_size;++index,++pixels)
        {
            if (fib_data.fib.getFA(index,order) == 0.0)
            {
                *pixels = 0;
                continue;
            }

            float fa = fib_data.fib.getFA(index,order)*255.0;
                        image::vector<3,float> dir = fib_data.fib.getDir(index,order);

            unsigned int color = (unsigned char)std::abs(dir[0]*fa);
            color <<= 8;
            color |= (unsigned char)std::abs(dir[1]*fa);
            color <<= 8;
            color |= (unsigned char)std::abs(dir[2]*fa);
            *pixels = color;
        }
    }
public:
    static unsigned int make_color(unsigned char gray)
    {
        unsigned int color = gray;
        color <<=8;
        color |= gray;
        color <<=8;
        color |= gray;
        return color;
    }
    void get_slice(const std::string& view_name,const std::string& overlay_name,
                   unsigned char dim,unsigned int pos,
                   image::color_image& show_image,float contrast,float offset)
    {
        unsigned int view_index = get_name_index(view_name);
        if(view_index == fib_data.view_item.size())
            return;

        if(fib_data.view_item[view_index].name == "color")
        {

            if(fib_data.view_item[view_index].data_buf.empty())
            {
                image::basic_image<image::rgb_color,3> color_buf(fib_data.dim);
                float max_value = fib_data.view_item[0].max_value;
                if(max_value + 1.0 == 1.0)
                    max_value = 1.0;
                float r = 255.9/max_value;
                for (unsigned int index = 0;index < fib_data.total_size;++index)
                {
                    image::vector<3,float> dir = fib_data.fib.getDir(index,0);
                                    dir *= std::floor(fib_data.fib.getFA(index,0)*r);
                    unsigned int color = (unsigned char)std::abs(dir[0]);
                    color <<= 8;
                    color |= (unsigned char)std::abs(dir[1]);
                    color <<= 8;
                    color |= (unsigned char)std::abs(dir[2]);
                    color_buf[index] = color;
                }
                color_buf.swap(fib_data.view_item[view_index].data_buf);
            }
            image::reslicing(fib_data.view_item[view_index].data_buf, show_image, dim,pos);
        }
        else
        {
            image::basic_image<float,2> buf;
            image::reslicing(fib_data.view_item[view_index].image_data, buf, dim, pos);
            show_image.resize(buf.geometry());
            float max_value = fib_data.view_item[view_index].max_value;
            float min_value = fib_data.view_item[view_index].min_value;
            float range = (max_value-min_value);
            if(range + 1.0 == 1.0)
                range = 1.0;

            buf += offset*range-min_value;
            buf *= 255.9*contrast/range;
            image::upper_lower_threshold(buf,(float)0.0,(float)255.0);
            std::copy(buf.begin(),buf.end(),show_image.begin());
        }


        if(overlay_name.length()) // has overlay
        {
            unsigned int overlay_index = get_name_index(overlay_name);
            if(overlay_index == fib_data.view_item.size())
                return;
            if(fib_data.view_item[overlay_index].data_buf.empty())
            {
                using namespace std;
                const float* data = fib_data.view_item[overlay_index].image_data.begin();
                float max_value = fib_data.view_item[overlay_index].max_value;
                float min_value = fib_data.view_item[overlay_index].min_value;
                float r = max(max_value,-min_value);
                if(r + 1.0 == 1.0)
                    r = 1.0;
                r = 255.9/r;
                image::basic_image<image::rgb_color,3> color_buf(fib_data.dim);
                for (unsigned int index = 0;index < fib_data.total_size;++index)
                    if(data[index] != 0.0)
                    {
                        image::rgb_color color;
                        color.g = std::floor(std::abs(data[index]*r));
                        if(data[index] > 0.0)
                            color.r = 255;
                        else
                            color.b = 255;
                        color_buf[index] = color;
                    }
                color_buf.swap(fib_data.view_item[overlay_index].data_buf);
            }
            image::color_image buf;
            image::reslicing(fib_data.view_item[overlay_index].data_buf, buf, dim, pos);
            for(unsigned int index = 0;index < buf.size();++index)
                if((unsigned int)(buf[index]) != 0)
                    show_image[index] = buf[index];
        }
    }

    void get_voxel_info2(unsigned int x,unsigned int y,unsigned int z,std::vector<float>& buf) const
    {
        unsigned int index = (z*fib_data.dim[1]+y)*fib_data.dim[0] + x;
        if (index >= fib_data.total_size)
            return;
        for(unsigned int fib = 0;fib < fib_data.fib.num_fiber;++fib)
        {
            image::vector<3,float> dir = fib_data.fib.getDir(index,fib);
            buf.push_back(dir[0]);
            buf.push_back(dir[1]);
            buf.push_back(dir[2]);
        }
    }
    void get_voxel_information(unsigned int x,unsigned int y,unsigned int z,std::vector<float>& buf) const
    {
        unsigned int index = (z*fib_data.dim[1]+y)*fib_data.dim[0] + x;
        if (index >= fib_data.total_size)
            return;
        for(unsigned int i = 0;i < fib_data.view_item.size();++i)
            if(fib_data.view_item[i].name != "color")
                buf.push_back(fib_data.view_item[i].image_data.empty() ? 0.0 : fib_data.view_item[i].image_data[index]);
    }
public:
    bool load_from_file(const char* file_name)
    {
        if(!fib_data.load_from_file(file_name))
            return false;
        vbc.reset(new vbc_database);
        if(!vbc->load_template(this))
            vbc.reset(0);
        return true;
    }


    void get_index_titles(std::vector<std::string>& titles)
    {
        if (fib_data.view_item[0].name[0] == 'f')// is dti
        {
            titles.push_back("FA mean");
            titles.push_back("FA sd");
        }
        else
        {
            titles.push_back("QA mean");
            titles.push_back("QA sd");
        }
        for(int data_index = fib_data.other_mapping_index;
            data_index < fib_data.view_item.size();++data_index)
        {
            titles.push_back(fib_data.view_item[data_index].name+" mean");
            titles.push_back(fib_data.view_item[data_index].name+" sd");
        }

    }


};





#endif//ODF_MODEL_HPP

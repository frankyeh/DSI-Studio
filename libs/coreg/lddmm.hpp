#include "image/image.hpp"
#include <boost/thread/thread.hpp>
#include <algorithm>

struct lddmm_param
{
    double image_ratio;
    float d_ratio;       // 0.02
    float v_decay_ratio; // 0.99999
	float delta_t;
    lddmm_param(void)
    {
        d_ratio = 0.8;
        v_decay_ratio = 0.98;
    }
};


template<unsigned int dimension>
class flow
{
public:
    const flow<dimension> *previous_flow;
    const flow<dimension> *counter_flow;
public:
    const lddmm_param& param;
    image::geometry<dimension> geo;
    image::basic_image<image::vector<dimension,float>,dimension> s;
    image::basic_image<short,dimension> j;
public:
    flow(const lddmm_param& param_,const image::geometry<dimension>& geo_):
            previous_flow(this),counter_flow(this),param(param_),geo(geo_),s(geo),j(geo) {}
    flow(const flow& rhs):
            previous_flow(&rhs),counter_flow(0),param(rhs.param),geo(rhs.geo),s(rhs.s),j(rhs.j) {}
public:
    void get_v(image::basic_image<image::vector<dimension,float>,dimension>& v)
    {
        double scaling = param.image_ratio;
        scaling *= param.image_ratio;

        image::basic_image<image::vector<dimension,float>,dimension> dv;
        image::gradient(j,dv);
        image::basic_image<short,dimension> dj(j);
		image::minus(dj.begin(),dj.end(),counter_flow->j.begin());
        image::multiply(dv.begin(),dv.end(),dj.begin());

        image::basic_image<int,dimension> D;
        image::jacobian_determinant(counter_flow->s,D);

		image::multiply(dv.begin(),dv.end(),D.begin());
		
		image::filter::gaussian(dv);
        image::filter::gaussian(dv);


        scaling *= param.d_ratio;
		image::multiply_constant(v.begin(),v.end(),param.v_decay_ratio);
		image::multiply_constant(dv.begin(),dv.end(),scaling);
		image::add(v.begin(),v.end(),dv.begin());
    }
    void update_s(const image::basic_image<image::vector<dimension,float>,dimension>& v,
                              const image::basic_image<image::vector<dimension,float>,dimension>& previous_v,
			      bool counter)
    {
		float delta_t = param.delta_t/2.0;
		if(counter)
		{
            for (image::pixel_index<dimension> index;index.valid(geo);index.next(geo))
            {
                image::vector<dimension,float> location(index);
                image::vector<dimension,float> dv(v[index.index()]);
                dv += previous_v[index.index()];
				dv *= delta_t;
                location += dv;
				image::linear_estimate(previous_flow->s,location,s[index.index()]);
            }
		}
		else
        {
            for (image::pixel_index<dimension> index;index.valid(geo);index.next(geo))
            {
                image::vector<dimension,float> location(index);
                image::vector<dimension,float> dv(v[index.index()]);
                dv += previous_v[index.index()];
                dv *= delta_t;
                location -= dv;
                image::linear_estimate(previous_flow->s,location,s[index.index()]);
            }
        }
    }
};

template<unsigned int dimension>
class time_frame
{
public:
    image::geometry<dimension> geo;
    image::basic_image<image::vector<dimension,float>,dimension> v;
	flow<dimension> flow0,flow1;
private:
    void setup_flow(void)
    {
        flow0.counter_flow = &flow1;
        flow1.counter_flow = &flow0;
    }
public:
    time_frame(const lddmm_param& param,const image::geometry<dimension>& geo_):
            geo(geo_),v(geo),flow0(param,geo_),flow1(param,geo_)
    {
        setup_flow();
    }

    time_frame(time_frame& previous_frame,time_frame& next_frame):
            geo(previous_frame.geo),flow0(previous_frame.flow0),flow1(next_frame.flow1)
    {
        previous_frame.flow1.previous_flow = &flow1;
        next_frame.flow0.previous_flow = &flow0;
		v = previous_frame.v;
		image::add(v.begin(),v.end(),next_frame.v.begin());
		image::multiply_constant(v.begin(),v.end(),0.5);
        setup_flow();
    }

    void set_identity_displacement(void)
    {
        for (image::pixel_index<dimension> index;index.valid(geo);index.next(geo))
            flow0.s[index.index()] = flow1.s[index.index()] = image::vector<dimension,float>(index);
    }



};

template<unsigned int dimension>
class LDDMM
{
public:
    lddmm_param param;
    std::vector<time_frame<dimension>*> lddmm;
	std::auto_ptr<boost::thread_group> threads;
	bool abort;
public:
    ~LDDMM(void)
    {
        for (unsigned int index = 0;index < lddmm.size();++index)
            delete lddmm[index];
    }
    void preprossing(image::basic_image<short,dimension>& from,image::basic_image<short,dimension>& to)
    {
        // translation

        {
            image::vector<dimension,int> displacement;
            image::vector<dimension,int> sum_from,sum_to;
            int value_from = 0;
            int value_to = 0;
            for (image::pixel_index<dimension> index;index.valid(from.geometry());index.next(from.geometry()))
            {
                image::vector<dimension,int> w(index);
                sum_from += w * from[index];
                sum_to += w * to[index];
                value_from += from[index];
                value_to += to[index];
            }
            sum_from /= value_from;
            sum_to /= value_to;
            displacement = sum_from - sum_to;
            displacement /= 2;
            image::move(to,displacement);
            displacement = -displacement;
            image::move(from,displacement);
        }

        // trim images
        {
            image::basic_image<short,dimension> from_,to_;
            image::basic_image<short,dimension> dif(from);
            dif -= to;
            image::geometry<dimension> crop_from,crop_to;
            image::trim(dif,crop_from,crop_to);
            if (crop_from[0] != crop_to[0] ||
                    crop_from[1] != crop_to[1])
            {
                int border = 10;
                for (unsigned int index = 0;index < dimension;++index)
                {
                    if (crop_from[index] < border)
                        crop_from[index] = 0;
                    else
                        crop_from[index] -= border;

                    if (crop_to[index] + border <= from.geometry()[index])
                        crop_to[index] += border;
                    else
                        crop_to[index] = from.geometry()[index];
                }
                image::crop(from,from_,crop_from,crop_to);
                image::crop(to,to_,crop_from,crop_to);
                from_.swap(from);
                to_.swap(to);
            }

        }
    }
    void init(const image::basic_image<short,dimension>& from,
              const image::basic_image<short,dimension>& to)
    {
        float max_value = std::max<short>((*std::max_element(from.begin(),from.end())),
                                   (*std::max_element(to.begin(),to.end())));
        param.image_ratio = 1.0/max_value;


        lddmm.push_back(new time_frame<dimension>(param,from.geometry()));
        lddmm.push_back(new time_frame<dimension>(param,from.geometry()));
        lddmm.front()->set_identity_displacement();
        lddmm.front()->flow0.j = from;
        lddmm.front()->flow1.j = from;
        lddmm.back()->set_identity_displacement();
        lddmm.back()->flow0.j = to;
        lddmm.back()->flow1.j = to;
    }

    void insert(void)
    {
        std::vector<time_frame<dimension>*> new_lddmm;
        for (unsigned int index = 0;index < lddmm.size()-1;++index)
        {
            new_lddmm.push_back(lddmm[index]);
            new_lddmm.push_back(new time_frame<dimension>(*lddmm[index],*lddmm[index+1]));
        }
        new_lddmm.push_back(lddmm.back());
        new_lddmm.swap(lddmm);
		param.delta_t = (float)1.0/((float)lddmm.size());	
    }

    void update(void)
    {
        {
            for (unsigned int index = 0;index < lddmm.size() && !abort;++index)
                lddmm[index]->flow0.get_v(lddmm[index]->v);
            for (unsigned int index = 1;index < lddmm.size() && !abort;++index)
            {
                lddmm[index]->flow0.update_s(lddmm[index]->v,lddmm[index-1]->v,false);
                image::compose_mapping(lddmm.front()->flow0.j,lddmm[index]->flow0.s,lddmm[index]->flow0.j);
            }
            for (int index = lddmm.size()-2;index >= 0 && !abort;--index)
            {
                lddmm[index]->flow1.update_s(lddmm[index]->v,lddmm[index+1]->v,true);
                image::compose_mapping(lddmm.back()->flow1.j,lddmm[index]->flow1.s,lddmm[index]->flow1.j);
            }
        }
    }
	void run(unsigned int frame_number,unsigned int iteration)
	{
		abort = false;
		while(lddmm.size() < frame_number && !abort)
		{
			insert();
			update();
		}
                for(unsigned int index = 0;index < iteration && !abort;++index)
			update();
	}
	void thread_run(unsigned int frame_number,unsigned int iteration)
	{
		threads.reset(new boost::thread_group);
		threads->add_thread(new boost::thread(&LDDMM<dimension>::run,this,frame_number,iteration));
	}
};

//------------------------------------------------------------------------------------
/*

template<int dimension>
class subject_map{
public:
	const lddmm_param& param;
    image::geometry<dimension> geo;
    image::basic_image<image::vector<dimension,float>,dimension> s;
    image::basic_image<short,dimension> j0,jn;
public:
	subject_map(const lddmm_param& param_,image::geometry<dimension> geo_,const short* image):
	  param(param_),geo(geo_),s(geo_),j0(geo_),jn(geo_)
	{
		std::copy(image,image+geo.size(),j0.begin());
		jn = j0;
		for (image::pixel_index<dimension> index;index.valid(geo));index.next(geo))
            s[index] = image::vector<dimension,float>(index);
	}
	void update(const image::basic_image<short,dimension>& jt)
	{
        double scaling = param.image_ratio;
        scaling *= param.image_ratio;

        image::basic_image<image::vector<dimension,float>,dimension> v;
        image::gradient(jn,v);
        image::basic_image<short,dimension> dj(jn);
        dj -= jt;
        v *= dj;

        
        image::gaussian_filter(v);

		
		scaling *= param.d_ratio;
		v *= scaling;
		
                image::basic_image<image::vector<dimension,float>,dimension> new_s(geo);
        for (image::pixel_index<dimension> index;index.valid(geo);index.next(geo))
        {
            image::vector<dimension,float> location(index);
            location -= v[index];
            image::estimate(s,location,new_s[index]);
        }
		s.swap(new_s);
		image::compose(j0,s,jn);
	}

};

template<int dimension>
class lddmm_map{

public:
    lddmm_param param;
    image::geometry<dimension> geo;
    std::vector<subject_map<dimension>*> lddmm;
	std::auto_ptr<boost::thread_group> threads;
    image::basic_image<short,dimension> j;
	short max_value;
	bool abort;
public:
	lddmm_map(const image::geometry<dimension>& geo_):geo(geo_),j(geo_),max_value(0){}
    ~lddmm_map(void)
	{
        for (unsigned int index = 0;index < lddmm.size();++index)
            delete lddmm[index];
    }
	void add_image(const short* image)
	{
		lddmm.push_back(new subject_map<dimension>(param,geo,image));
                unsigned int count = 0;
		int value = 0;
                for(unsigned int index = 0;index < j.size();++index)
			if(image[index])
			{
				value += std::abs(image[index]);
			    ++count;
			}
		float average = value;
		average /= count;
		if(average > max_value)
			    max_value = average;
	}
	void run(unsigned int iteration)
	{
		param.image_ratio = 1.0/((float)max_value);
                for(unsigned int i = 0;i < iteration;++i)
		{
			image::basic_image<int,dimension> j_sum(geo);
                        for(unsigned int index = 0;index < lddmm.size();++index)
				j_sum += lddmm[index]->jn;
			j_sum /= (float)lddmm.size();
			j.resize(geo);
			std::copy(j_sum.begin(),j_sum.end(),j.begin());
                        for(unsigned int index = 0;index < lddmm.size();++index)
				lddmm[index]->update(j);
		}
	}

	void thread_run(unsigned int iteration)
	{
		threads.reset(new boost::thread_group);
		threads->add_thread(new boost::thread(&lddmm_map<dimension>::run,this,iteration));
	}

};

*/


template<int dimension>
class lddmm_map{

public:
    image::geometry<dimension> geo;
    std::vector<LDDMM<dimension>*> lddmms;
	std::auto_ptr<boost::thread_group> threads;
    image::basic_image<short,dimension> j;
	short max_value;
	bool abort;
public:
	lddmm_map(const image::geometry<dimension>& geo_):geo(geo_),max_value(0){}
    ~lddmm_map(void)
	{
        for (unsigned int index = 0;index < lddmms.size();++index)
            delete lddmms[index];
    }
	void add_image(const short* image)
	{
		if(lddmms.empty())
                    j = image::basic_image<short,dimension>(image,geo);
		lddmms.push_back(new LDDMM<dimension>());
		lddmms.back()->init(j,image::basic_image<short,dimension>(image,geo));
	}
	void update_template(void)
	{
                image::basic_image<image::vector<dimension,float>,dimension> vt(geo);
                for(unsigned int index = 0;index < lddmms.size() && !abort;++index)
		    image::add(vt.begin(),vt.end(),lddmms[index]->lddmm.front()->v.begin());
		image::divide_constant(vt.begin(),vt.end(),(float)lddmms.size());	
	}
	void run(unsigned int frame_number,unsigned int iteration)
	{
		abort = false;
		while(lddmms.size() < frame_number && !abort)
		{
                        for(unsigned int index = 0;index < lddmms.size() && !abort;++index)
				lddmms[index]->insert();
                        for(unsigned int index = 0;index < lddmms.size() && !abort;++index)
				lddmms[index]->update();
			update_template();

		}
                for(unsigned int i = 0;i < iteration && !abort;++i)
		{
                        for(unsigned int index = 0;index < lddmms.size() && !abort;++index)
				lddmms[index]->update();
			update_template();
			
		}
	}

	void thread_run(unsigned int frame_number,unsigned int iteration)
	{
		threads.reset(new boost::thread_group);
		threads->add_thread(new boost::thread(&lddmm_map<dimension>::run,this,frame_number,iteration));
	}

};

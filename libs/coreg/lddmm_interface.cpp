#include "lddmm.hpp"
extern "C"
    void* create_lddmm2(unsigned int width,unsigned int height,short* from_image,short* to_image)
{
	image::basic_image<short,2> from(image::geometry<2>(width,height)),to(image::geometry<2>(width,height));
	std::copy(from_image,from_image+from.size(),from.begin());
	std::copy(to_image,to_image+to.size(),to.begin());
	

	
    std::auto_ptr<LDDMM<2> > lddmm(new LDDMM<2>());
    lddmm->init(from,to);

    return lddmm.release();
}

extern "C"
unsigned int lddmm2_get_frame_count(LDDMM<2>* lddmm)
{
	return lddmm->lddmm.size();
}

extern "C"
void lddmm2_run(LDDMM<2>* lddmm,unsigned int frame_number,unsigned int iteration)
{
    lddmm->thread_run(frame_number,iteration);
}

extern "C"
const short* lddmm2_getj(LDDMM<2>* lddmm,unsigned int index,bool to)
{
	return to ? &*(lddmm->lddmm[index]->flow0.j.begin()) : &*(lddmm->lddmm[index]->flow1.j.begin());
}
extern "C"
const short* lddmm2_getv(LDDMM<2>* lddmm,unsigned int i)
{
	image::basic_image<float,2> j(lddmm->lddmm.front()->flow0.j.geometry());
        for (unsigned int index = 0;index < j.size();++index)
		j[index] = lddmm->lddmm[i]->v[index].length();
	
	float m = *std::max_element(j.begin(),j.end());
        if (m == 0.0)
            m = 1.0;
        float ratio = 254.0/m;
	image::multiply_constant(j.begin(),j.end(),ratio);
	static image::basic_image<short,2> return_image(j.geometry());
	std::copy(j.begin(),j.end(),return_image.begin());
	return &*return_image.begin();
}

extern "C"
void lddmm2_free(LDDMM<2>* lddmm)
{
	delete lddmm;
}



//--------------------------------------------------------------------------
extern "C"
    void* create_lddmm3(unsigned int width,unsigned int height,unsigned int depth,short* from_image,short* to_image)
{
	image::basic_image<short,3> from(image::geometry<3>(width,height,depth)),to(image::geometry<3>(width,height,depth));
	std::copy(from_image,from_image+from.size(),from.begin());
	std::copy(to_image,to_image+to.size(),to.begin());
	

	
    std::auto_ptr<LDDMM<3> > lddmm(new LDDMM<3>());
    lddmm->init(from,to);

    return lddmm.release();
}

extern "C"
unsigned int lddmm3_get_frame_count(LDDMM<3>* lddmm)
{
	return lddmm->lddmm.size();
}

extern "C"
void lddmm3_run(LDDMM<3>* lddmm,unsigned int frame_number,unsigned int iteration)
{
    lddmm->thread_run(frame_number,iteration);
}

extern "C"
const short* lddmm3_getj(LDDMM<3>* lddmm,unsigned int index,bool to)
{
	return to ? &*(lddmm->lddmm[index]->flow0.j.begin()) : &*(lddmm->lddmm[index]->flow1.j.begin());
}
extern "C"
const short* lddmm3_getv(LDDMM<3>* lddmm,unsigned int i)
{
	image::basic_image<float,3> j(lddmm->lddmm.front()->flow0.j.geometry());
        for (unsigned int index = 0;index < j.size();++index)
		j[index] = lddmm->lddmm[i]->v[index].length();

	
	float m = *std::max_element(j.begin(),j.end());
        if (m == 0.0)
            m = 1.0;
        float ratio = 254.0/m;
	image::multiply_constant(j.begin(),j.end(),ratio);
	static image::basic_image<short,3> return_image(j.geometry());
	std::copy(j.begin(),j.end(),return_image.begin());
	return &*return_image.begin();
}

extern "C"
void lddmm3_free(LDDMM<3>* lddmm)
{
	delete lddmm;
}

//-----------------------------------------------------------------------------


extern "C"
    void* create_lddmm2_map(unsigned int width,unsigned int height)
{
	std::auto_ptr<lddmm_map<2> > lddmm(new lddmm_map<2>(image::geometry<2>(width,height)));
    return lddmm.release();
}


extern "C"
void lddmm2_map_add_image(lddmm_map<2>* lddmm,const short* image)
{
	lddmm->add_image(image);
}

extern "C"
void lddmm2_map_run(lddmm_map<2>* lddmm,unsigned int iteration)
{
    lddmm->thread_run(20,iteration);
}

extern "C"
const short* lddmm2_map_getj(lddmm_map<2>* lddmm)
{
	return &*(lddmm->j.begin());
}

extern "C"
void lddmm2_map_free(lddmm_map<2>* lddmm)
{
	delete lddmm;
}

//-----------------------------------------------------------------------------

extern "C"
    void* create_lddmm3_map(unsigned int width,unsigned int height,unsigned int depth)
{
	std::auto_ptr<lddmm_map<3> > lddmm(new lddmm_map<3>(image::geometry<3>(width,height,depth)));
    return lddmm.release();
}


extern "C"
void lddmm3_map_add_image(lddmm_map<3>* lddmm,const short* image)
{
	lddmm->add_image(image);
}

extern "C"
void lddmm3_map_run(lddmm_map<3>* lddmm,unsigned int iteration)
{
    lddmm->thread_run(20,iteration);
}

extern "C"
const short* lddmm3_map_getj(lddmm_map<3>* lddmm)
{
	return &*(lddmm->j.begin());
}

extern "C"
void lddmm3_map_free(lddmm_map<3>* lddmm)
{
	delete lddmm;
}

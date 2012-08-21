#ifndef GAUSSIAN_FITTING_HPP
#define GAUSSIAN_FITTING_HPP
#include <vector>
#include <cmath>
#include "tessellated_icosahedron.hpp"
#include "math/root.hpp"
#include "odf_decomposition.hpp"

class ShapingModel : public SaveFiberInfo
{
private:
    ConvolutedOdfComponent odf_component;
public:
    virtual void init(Voxel& voxel)
    {

        SaveFiberInfo::init(voxel);
        odf_component.icosa_components.resize(voxel.ti.half_vertices_count);
        for(unsigned int index = 0; index < voxel.ti.half_vertices_count; ++index)
            odf_component.icosa_components[index].initialize(index);
    }

    virtual void run(Voxel& voxel, VoxelData& data)
    {

        std::vector<float> odf(data.odf);
        float min_odf = RemoveIsotropicPart()(odf);

        for(unsigned int index = 0; index < max_odf_record; ++index)
        {
            odf_component.decomposeODF(odf);
            unsigned short mv = odf_component.getMainVector();
            findex[index][data.voxel_index] = mv;
            fa[index][data.voxel_index] = odf_component.getExtractedODF()[mv];
            //	min_odf += odf_component.getExtractedODF()[mv];
        }
        //if(min_odf == 0.0)
        //	min_odf = 1.0;
        //for(unsigned int index = 0;index < max_odf_record;++index)
        //	fa[index][data.voxel_index] /= min_odf;


        /*odf = old_odf;
        RemoveIsotropicPart()(odf);

        odf_component[0].decomposeODF(odf,voxel_info.index[1]);
        unsigned int max_index = std::max_element(odf.begin(),odf.end())-odf.begin();
        bool tag = std::abs(image::vector<3,float>(ti_vertices(voxel_info.index[0]))*
        //	image::vector<3,float>(ti_vertices(voxel_info.index[1])))
        //	> 0.8;
        if(max_index != voxel_info.index[0])
        {
        	odf = old_odf;
        	RemoveIsotropicPart()(odf);

        	odf_component[0].decomposeODF(odf,max_index);
        	voxel_info.index[0] = max_index;
        	voxel_info.fa[0] = odf_component[0].getExtractedODF()[max_index]/
        					(odf_component[0].getExtractedODF()[max_index]+min_odf);

        	odf_component[1].decomposeODF(odf);
        	voxel_info.index[1] = odf_component[1].getMainVector();
        	voxel_info.fa[1] = odf_component[1].getExtractedODF()[voxel_info.index[1]]/
        					(odf_component[1].getExtractedODF()[voxel_info.index[1]]+min_odf);

        	odf_component[2].decomposeODF(odf);
        	voxel_info.index[2] = odf_component[2].getMainVector();
        	voxel_info.fa[2] = odf_component[2].getExtractedODF()[voxel_info.index[2]]/
        					(odf_component[2].getExtractedODF()[voxel_info.index[2]]+min_odf);

        }
        */
    }
};
/*
class ConvolutedShapingModel{


private:
	OdfComponent odf_component[max_odf_record];
public:
        const OdfComponent& operator[](unsigned int index) const{return odf_component[index];}
	void operator()(std::vector<float>& old_odf)
	{
		std::vector<float> odf(old_odf.get());
   	    float min_odf = RemoveIsotropicPart()(odf);

                unsigned int index = 0;
                for(unsigned int index = 0;index < max_odf_record;++index)
		{
			odf_component[index].decomposeODF(odf);
			std::vector<float> re_odf(odf_size);
                        for(unsigned int i = 0;i < odf_size;++i)
				re_odf[i] = min_odf + odf_component[index].getExtractedODF()[i];

			old_voxel_info.push(odf_component[index].getMainVector(),GeneralizedFA()(re_odf));
		}
	}


};
*/
#endif//GAUSSIAN_FITTING_HPP

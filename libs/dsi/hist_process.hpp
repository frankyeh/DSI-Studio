#ifndef HIST_PROCESS_HPP
#define HIST_PROCESS_HPP
#include "basic_process.hpp"
#include "basic_voxel.hpp"

class ReadImages : public BaseProcess{

    virtual void run_hist(Voxel& voxel,HistData& hist)
    {
        // crop from original image
        tipl::crop(voxel.hist_image,hist.I,hist.from,hist.to);

        // translate mask from downsampled space to the original space
        auto& mask = hist.I_mask;
        mask.resize(hist.I.shape());

        float rx = float(voxel.mask.width()-1)/float(voxel.hist_image.width()-1);
        float ry = float(voxel.mask.height()-1)/float(voxel.hist_image.height()-1);

        for(tipl::pixel_index<2> p(mask.shape());p < mask.size();++p)
        {
            tipl::vector<2> pos(p);
            pos += hist.from;
            pos[0] *= rx;
            pos[1] *= ry;
            if(voxel.mask.shape().is_valid(int(pos[0]),int(pos[1]),0))
                hist.I_mask[p.index()] = voxel.mask.at(uint32_t(pos[0]),uint32_t(pos[1])) ? 255:0;
        }
    }

};

class CalculateGradient : public BaseProcess{

    virtual void run_hist(Voxel& voxel,HistData& hist)
    {
        for(unsigned int i = 0;i < voxel.hist_raw_smoothing;++i)
            tipl::filter::gaussian(hist.I);
        tipl::image<2> dx,dy;
        tipl::gradient_2x(hist.I,dx);
        tipl::gradient_2y(hist.I,dy);

        hist.other_maps[HistData::dx] = std::move(dx);
        hist.other_maps[HistData::dy] = std::move(dy);
    }
};

class CalculateStructuralTensor : public BaseProcess{

    virtual void run_hist(Voxel& voxel,HistData& hist)
    {
        tipl::image<2> dxx,dyy,dxy;
        dxx = std::move(hist.other_maps[HistData::dx]);
        dyy = std::move(hist.other_maps[HistData::dy]);
        dxy = dxx;
        dxy *= dyy;
        tipl::square(dxx);
        tipl::square(dyy);

        for(unsigned int i = 0;i < voxel.hist_downsampling;++i)
        {
            tipl::downsample_with_padding(dxx);
            tipl::downsample_with_padding(dyy);
            tipl::downsample_with_padding(dxy);
            tipl::downsample_with_padding(hist.I_mask);
        }

        for(unsigned int i = 0;i < voxel.hist_tensor_smoothing;++i)
        {
            tipl::filter::gaussian(dxx);
            tipl::filter::gaussian(dyy);
            tipl::filter::gaussian(dxy);
        }
        // apply mask here
        for(size_t i = 0;i < hist.I_mask.size();++i)
            if(!hist.I_mask[i])
            {
                dxx[i] = 0.0f;
                dyy[i] = 0.0f;
                dxy[i] = 0.0f;
            }

        tipl::vector<2,int> from(0,0),to(dxx.width(),dxx.height());
        // remove left and upper borders
        if(hist.from[0] > voxel.margin)
        {
            from[0] += (voxel.margin >> voxel.hist_downsampling);
            hist.from[0] += voxel.margin;
        }
        if(hist.from[1] > voxel.margin)
        {
            from[1] += (voxel.margin >> voxel.hist_downsampling);
            hist.from[1] += voxel.margin;
        }
        // limit image size to crop_size
        to[0] = std::min<int>(to[0],from[0]+(voxel.crop_size >> voxel.hist_downsampling));
        to[1] = std::min<int>(to[1],from[1]+(voxel.crop_size >> voxel.hist_downsampling));

        tipl::crop(dxx,hist.other_maps[HistData::dxx],from,to);
        tipl::crop(dyy,hist.other_maps[HistData::dyy],from,to);
        tipl::crop(dxy,hist.other_maps[HistData::dxy],from,to);

    }
};

class EigenAnalysis: public BaseProcess{
    tipl::image<3> hist_fa;
    tipl::image<3,tipl::vector<3> > hist_dir;
public:
    virtual void init(Voxel& voxel)
    {
        tipl::vector<3> new_vs(voxel.vs);
        tipl::shape<3> new_dim(voxel.hist_image.width(),voxel.hist_image.height(),1);
        for(unsigned int i = 0;i < voxel.hist_downsampling;++i)
        {
            new_vs *= 2.0f;
            new_dim[0] = (new_dim[0]+1) >> 1;
            new_dim[1] = (new_dim[1]+1) >> 1;
        }
        hist_fa.resize(new_dim);
        hist_dir.resize(new_dim);
        new_dim[2] = 2;
        voxel.dim = new_dim;
        voxel.vs = new_vs;
    }
    virtual void run_hist(Voxel& voxel,HistData& hist)
    {
        tipl::image<2> dxx,dyy,dxy;
        dxx = std::move(hist.other_maps[HistData::dxx]);
        dyy = std::move(hist.other_maps[HistData::dyy]);
        dxy = std::move(hist.other_maps[HistData::dxy]);

        tipl::image<3> hist_fa_sub({dxx.width(),dxx.height(),1});
        tipl::image<3,tipl::vector<3> > hist_dir_sub({dxx.width(),dxx.height(),1});

        for(size_t i = 0;i < dxx.size();++i)
            if(dxx[i] != 0.0f)
            {
                float dxx_ = dxx[i];
                float dyy_ = dyy[i];
                float dxy_ = dxy[i];
                float trace = dxx_+dyy_;
                float det = dxx_*dyy_-dxy_*dxy_;
                float deta = std::sqrt(trace*trace-4.0f*det);
                float nu2 = 0.5f*(trace+deta);
                hist_fa_sub[i] = std::log(nu2+1.0f);
                hist_dir_sub[i] = {dxy_,dyy_-nu2,0.0f};
                hist_dir_sub[i].normalize();
            }
        auto draw_location = tipl::vector<3>(hist.from[0] >> voxel.hist_downsampling,
                                             hist.from[1] >> voxel.hist_downsampling,0);
        tipl::draw(hist_fa_sub,hist_fa,draw_location);
        tipl::draw(hist_dir_sub,hist_dir,draw_location);
    }
    virtual void end(Voxel&,tipl::io::gz_mat_write& mat_writer)
    {
        tipl::normalize(hist_fa);

        // create additional layer
        hist_fa.resize(tipl::shape<3>(hist_fa.width(),hist_fa.height(),2));
        std::copy(hist_fa.begin(),hist_fa.begin()+int64_t(hist_fa.plane_size()),hist_fa.begin()+int64_t(hist_fa.plane_size()));

        hist_dir.resize(tipl::shape<3>(hist_dir.width(),hist_dir.height(),2));
        std::copy(hist_dir.begin(),hist_dir.begin()+int64_t(hist_dir.plane_size()),hist_dir.begin()+int64_t(hist_dir.plane_size()));

        write_image_to_mat(mat_writer,"fa0",hist_fa.data(),hist_fa.shape());
        write_image_to_mat(mat_writer,"dir0",&hist_dir[0][0],hist_dir.shape().multiply(tipl::shape<3>::x,3));
    }
};



#endif // HIST_PROCESS_HPP

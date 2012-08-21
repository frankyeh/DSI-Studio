#ifndef MIX_GAUSSIAN_MODEL_HPP
#define MIX_GAUSSIAN_MODEL_HPP
#include <cmath>
#include <image/image.hpp>
class GaussianModel
{
private:
    float lambda1,lambda2;
    image::vector<3,float> axis;
public:
    GaussianModel(float l1,float l2,const image::vector<3,float>& axis_):
        lambda1(l1),lambda2(l2),axis(axis_) {}

    float operator()(float b,const image::vector<3,float>& g_dir) const
    {
        float cos2 = axis*g_dir;
        cos2*= cos2;
        return std::exp(-b*(cos2*(lambda1-lambda2)+lambda2));
    }

};


class MixGaussianModel
{
private:
    GaussianModel g1,g2;
    float f1,f2,f3;
    float iso_l;
public:
    MixGaussianModel(float l1,float l2,float l0,float angle,float fraction1,float fraction2):
        g1(l1,l2,image::vector<3,float>(1.0,0.0,0.0)),
        g2(l1,l2,image::vector<3,float>(std::cos(angle),std::sin(angle),0.0)),
        f1(fraction1),
		f2(fraction2),
		f3(1.0-fraction1-fraction2),
		iso_l(l0) {}

    float operator()(float b,const image::vector<3,float>& g_dir) const
    {
        return f1*g1(b,g_dir)+f2*g2(b,g_dir)+f3*std::exp(-b*iso_l);
    }

};




#endif//MIX_GAUSSIAN_MODEL_HPP

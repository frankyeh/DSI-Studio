#ifndef RACIAN_NOISE_HPP
#define RACIAN_NOISE_HPP
#include <map>
#include <cmath>
#include <ctime>
#include "tipl/tipl.hpp"

double modified_bessel_order0(double x)
{
        double y = std::abs(x);
        if(y < 3.75)
        {
                y /= 3.75;
                y *= y;
                return 1.0 + y * (3.5156229 + y * (3.0899424 +
                                         y * (1.2067492 + y * (0.2659732 +
                                         y * (0.360768e-1 + y * 0.45813e-2)))));
        }
        else
        {
                double z = 3.75 / y;
                return (std::exp(y)/std::sqrt(y))*(0.39894228 +
                        z * (0.1328592e-1 + z * (0.225319e-2 + z * (-0.157565e-2 +
                        z * (0.916281e-2   + z * (-0.2057706e-1 + z * (0.2635537e-1 +
                        z * (-0.1647633e-1 + z * 0.392377e-2))))))));
        }

}

class RacianNoise
{
public:
    float A;
    float nsd;
private:
    bool is_gaussian;
    float mean;
private:
    float max_prob;
    float max_int;
private:
    static tipl::normal_dist<float> gen_normal;
    static tipl::uniform_dist<float> gen_uniform;
private:
    struct RicianDistribution
    {
        float A;
        float A2;
        float nvr;
        RicianDistribution(float A_,float nsd):A(A_),A2(A_*A_),nvr(nsd*nsd) {}
        float operator()(float x) const
        {
            return x*std::exp(-0.5*(x*x+A2)/nvr)*modified_bessel_order0(A*x/nvr)/nvr;
        }
    } rician_dis;
public:
    RacianNoise(float A_,float nsd_):A(A_),nsd(nsd_),rician_dis(A_,nsd_)
    {
        const float prob_precision = 0.01f;
        // if snr > 3.0 the racian distribution is similar to Gaussian distribution
        if (A/nsd >= 3.0)
        {
            mean = std::sqrt(A*A+nsd*nsd);
            is_gaussian = true;
        }
        else
        {
            max_prob = 0.0;
            float x = 0.0;
            for (; 1; x += 0.2f)
            {
                float prob = rician_dis(x);
                if (prob < max_prob)
                    break;
                max_prob = prob;
            }

            x*= 2.0;
            for (float interval = x/4.0; 1; x += interval)
                if (rician_dis(x) < prob_precision*max_prob)
                    break;
            max_int = x;
            is_gaussian = false;
        }
    }

    float operator()(void)
    {
        float value;
        do
        {
            if (is_gaussian)
            {
                value = gen_normal()*nsd + mean;
            }
            else
            {
                do
                {
                    value = gen_uniform()*max_int;
                    float prob = rician_dis(value);
                    if (gen_uniform()*max_prob <= prob)
                        break;
                }
                while (1);
            }
        }
        while (value < 0.0);
        return value;
    }
};

#endif//RACIAN_NOISE_HPP

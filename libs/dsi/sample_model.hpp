#ifndef SAMPLE_MODEL_HPP
#define SAMPLE_MODEL_HPP

class SamplePoint
{
private:
    std::vector<float> ratio;
    std::vector<unsigned int> sample_index;
    unsigned int odf_index;
public:
    SamplePoint(void):ratio(8),sample_index(8),odf_index(0) {}
    SamplePoint(unsigned int odf_index_,float x,float y,float z,float weighting);
    void sampleODFValueWeighted(const std::vector<float>& pdf,std::vector<float>& odf) const;

};
#endif//SAMPLE_MODEL_HPP

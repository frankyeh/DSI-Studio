class ImageModel;
extern "C"
{
    // dsi
    void* init_reconstruction(const char* file_name);
    void free_reconstruction(ImageModel* image_model);
    const char* reconstruction(ImageModel* image_model,
                       unsigned int method_id,
                       const float* param_values);
    bool odf_average(const char* out_name,
            const char* const * file_name,unsigned int file_num);
    bool generate_simulation(
        const char* bvec_file_name,unsigned char s0_snr,float mean_dif,unsigned char odf_fold,
        const char* fa_iteration,
        const char* crossing_angle_iteration,
        unsigned char repeat_num);

}

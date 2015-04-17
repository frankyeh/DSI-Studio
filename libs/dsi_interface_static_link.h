class ImageModel;
const char* reconstruction(ImageModel* image_model,
                   unsigned int method_id,
                   const float* param_values,
                   bool check_btable,
                   unsigned int thread_count);
const char* odf_average(const char* out_name,
        const char* const * file_name,unsigned int file_num);

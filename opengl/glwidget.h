#ifndef GLWIDGET_H
#define GLWIDGET_H
#define NOMINMAX
#include <memory>
#include "QtOpenGL/QGLWidget"
#include "SliceModel.h"
#include "boost/ptr_container/ptr_vector.hpp"
#include "tracking/region/RegionModel.h"
#include "libs/coreg/linear.hpp"
#include "tracking/tracking_window.h"
class RenderingTableWidget;
 class GLWidget : public QGLWidget
 {
     Q_OBJECT

 public:
     GLWidget(bool samplebuffer,
              tracking_window& cur_tracking_window_,
              RenderingTableWidget* renderWidget_,
              QWidget *parent = 0);
     ~GLWidget();
 public:// editing
     unsigned char editing_option;
     float mat[16];
     image::vector<3,float> pos,dir1,dir2;
     void set_view(unsigned char view_option);
 private:
     bool object_selected,slice_selected;
     image::vector<3,float> accumulated_dis;
     float slice_distance;
     unsigned char moving_at_slice_index;
     float slice_dx,slice_dy;
     void slice_location(unsigned char dim,std::vector<image::vector<3,float> >& points);
     void get_view_dir(QPoint p,image::vector<3,float>& dir);
     void select_slice(void);
 private:
     int selected_index;
     float object_distance;
     void select_object(void);
 public:// other slices
     boost::ptr_vector<CustomSliceModel> other_slices;
     boost::ptr_vector<LinearMapping<image::basic_image<float,3,image::const_pointer_memory<float> >,image::rigid_scaling_transform<3> > > mi3s;
     std::vector<std::vector<float> > transform;
     unsigned int current_visible_slide;
     bool addSlices(QStringList filenames);
     void delete_slice(int index);
     const image::geometry<3>& getCurrentGeo(void) const
     {return other_slices[current_visible_slide-1].geometry;}
     void get_current_slice_transformation(image::geometry<3>& geo,
                                           image::vector<3,float>& vs,
                                           std::vector<float>& tr);
 private://surface
     std::auto_ptr<RegionModel> surface;

 private://odf
     std::vector<image::vector<3,float> >odf_points;
     std::vector<float>odf_colors;
     unsigned char odf_dim;
     unsigned char odf_slide_pos;
     void add_odf(int x,int y,int z);

 public slots:
     void makeTracts(void);
     void addSurface(void);
     void catchScreen(void);
     void saveCamera(void);
     void loadCamera(void);
     void saveRotationSeries(void);
     void saveLeftRight3DImage(void);
     void saveMapping(void);
     void loadMapping(void);
     void copyToClipboard(void);
 signals:
     void edited(void);
     void region_edited(void);
 protected:
     void initializeGL();
     void paintGL();
     void resizeGL(int width, int height);
     void mousePressEvent(QMouseEvent *event);
     void mouseReleaseEvent(QMouseEvent *event);
     void mouseMoveEvent(QMouseEvent *event);
     void mouseDoubleClickEvent(QMouseEvent *event);
     void wheelEvent ( QWheelEvent * event );
 private:
     tracking_window& cur_tracking_window;
     RenderingTableWidget* renderWidget;
     int width,height;
     float max_fa;

 private:
     int get_param(const char* name);
 private:
     unsigned char tract_alpha;
     unsigned char tract_style;
     unsigned char tract_color_style;
     unsigned char tube_size;
     unsigned char tract_color_contrast;
     unsigned char tract_tube_detail;
     unsigned char end_point_shift;
     unsigned char slice_contrast;
     unsigned char slice_offset;
     unsigned char slice_index;
 private:
     GLuint tracts,slice_texture[3];
     QPoint lastPos;
     float transformation_matrix[16];
     float scaled_factor;
     bool set_view_flip;

 };

#endif // GLWIDGET_H

#ifndef GLWIDGET_H
#define GLWIDGET_H
#include <QTimer>
#include <QTime>
//#include <QOpenGLShaderProgram>
#define NOMINMAX
#include <memory>
#include "QtOpenGL/QGLWidget"
#include <gl/GLU.h>
#include "tracking/region/RegionModel.h"
#include "tracking/tracking_window.h"
class RenderingTableWidget;
class GluQua{
private:
    GLUquadricObj* ptr;
public:
    GluQua(void):ptr(gluNewQuadric())
    {
    }
    ~GluQua(void)
    {
        gluDeleteQuadric(ptr);
    }
    GLUquadricObj* get(void) {return ptr;}
};

class GLWidget : public QGLWidget
{
Q_OBJECT

 public:
     GLWidget(bool samplebuffer,
              tracking_window& cur_tracking_window_,
              RenderingTableWidget* renderWidget_,
              QWidget *parent = 0);
     ~GLWidget();
public:
     //std::shared_ptr<QOpenGLShaderProgram> shader,shader2;
     //int s_mvp,s_mvp2,s_depthMap;
 public:// editing
     enum {none = 0,selecting = 1, moving = 2, dragging = 3} editing_option;
     tipl::vector<3,float> pos,dir1,dir2;
     std::vector<tipl::vector<3,float> > dirs;
     bool angular_selection;
     void get_pos(void);
     void set_view(unsigned char view_option);
     void scale_by(float scale);
     void move_by(int x,int y);
 private:
     bool object_selected,slice_selected;
     tipl::vector<3,float> accumulated_dis;
     float slice_distance;
     unsigned char moving_at_slice_index;
     float slice_dx,slice_dy;
     void slice_location(unsigned char dim,std::vector<tipl::vector<3,float> >& points);
     float get_slice_projection_point(unsigned char dim,
                                      const tipl::vector<3,float>& pos,const tipl::vector<3,float>& dir,
                                      float& dx,float& dy);
     void get_view_dir(QPoint p,tipl::vector<3,float>& dir);
     void select_slice(void);
 private:
     int selected_index;
     float object_distance;
     void select_object(void);
 public:// other slices
     QTime time;
     int last_time;
     bool get_mouse_pos(QMouseEvent *mouseEvent,tipl::vector<3,float>& position);
     void paintGL();

 public://surface
     std::auto_ptr<RegionModel> surface;

 private://odf
     std::vector<tipl::vector<3,float> >odf_points;
     std::vector<float>odf_colors;
     int odf_dim;
     int odf_slide_pos;
     void add_odf(tipl::pixel_index<3> pos);
private: //glu
     std::shared_ptr<GluQua> RegionSpheres;
public:
     tipl::image<float,2> connectivity;
 private:
     void rotate_angle(float angle,float x,float y,float z);
 public slots:
     void makeTracts(void);
     void addSurface(void);
     void catchScreen(void);
     void catchScreen2(void);
     void saveCamera(void);
     void loadCamera(void);
     void saveRotationSeries(void);
     void save3ViewImage(void);
     void copyToClipboard(void);
     void rotate(void);
 signals:
     void edited();
     void region_edited();
 protected:
     void initializeGL();
     void resizeGL(int width, int height);
     void setFrustum(void);
     void renderLR(void);
protected:
     bool edit_right;
     QPoint convert_pos(QMouseEvent *event);
     void mousePressEvent(QMouseEvent *event);
     void mouseReleaseEvent(QMouseEvent *event);
     void mouseMoveEvent(QMouseEvent *event);
     void mouseDoubleClickEvent(QMouseEvent *event);
     void wheelEvent ( QWheelEvent * event );
 private:
     tracking_window& cur_tracking_window;
     RenderingTableWidget* renderWidget;
     int cur_width,cur_height;
 private:
     int get_param(const char* name);
     float get_param_float(const char* name);
     bool check_change(const char* name,unsigned char& var);
     bool check_change(const char* name,float& var);
 private:
     float tract_alpha;
     unsigned char scale_voxel;
     unsigned char tract_alpha_style;
     unsigned char tract_style;
     unsigned char tract_color_style;
     float tube_diameter;
     unsigned char tract_color_contrast;
     unsigned char tract_tube_detail;
     unsigned char tract_variant_size;
     unsigned char tract_variant_color;
     unsigned char end_point_shift;
     unsigned char odf_position;
     unsigned char odf_skip;
     unsigned char odf_color;
     float odf_scale;
 public:
     GLuint tracts,slice_texture[3];
     int slice_pos[3];
     QPoint lastPos,last_select_point;
     tipl::matrix<4,4,float> mat,transformation_matrix,transformation_matrix2,rotation_matrix,rotation_matrix2;
     enum class view_mode_type { single, two, stereo} view_mode;

     bool set_view_flip;
     void get3View(QImage& I,unsigned int type);
     bool command(QString cmd,QString param = "",QString param2 = "");
 };

#endif // GLWIDGET_H

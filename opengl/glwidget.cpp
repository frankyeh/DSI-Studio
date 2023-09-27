#include <QtOpenGL>
#include <QtGui>
#include <QMessageBox>
#include <QInputDialog>
#include <QFileDialog>
#include <QSettings>
#include <QTimer>
#include <QClipboard>
#include <vector>
#include "glwidget.h"
#include "tracking/tracking_window.h"
#include "ui_tracking_window.h"
#include "renderingtablewidget.h"
#include "tracking/region/regiontablewidget.h"
#include "tracking/devicetablewidget.h"
#include "SliceModel.h"
#include "fib_data.hpp"
#include "tracking/color_bar_dialog.hpp"
#include "libs/tracking/tract_model.hpp"
#include "odf_process.hpp"


extern GLenum BlendFunc1[8],BlendFunc2[8];

GLenum BlendFunc1[8] = {GL_ZERO,GL_ONE,GL_DST_COLOR,
                      GL_ONE_MINUS_DST_COLOR,GL_SRC_ALPHA,
                      GL_ONE_MINUS_SRC_ALPHA,GL_DST_ALPHA,
                      GL_ONE_MINUS_DST_ALPHA};
GLenum BlendFunc2[8] = {GL_ZERO,GL_ONE,GL_SRC_COLOR,
                      GL_ONE_MINUS_DST_COLOR,GL_SRC_ALPHA,
                      GL_ONE_MINUS_SRC_ALPHA,GL_DST_ALPHA,
                      GL_ONE_MINUS_DST_ALPHA};

GLWidget::GLWidget(tracking_window& cur_tracking_window_,
                   RenderingTableWidget* renderWidget_,
                   QWidget *parent) : QOpenGLWidget(parent),
        cur_tracking_window(cur_tracking_window_),
        renderWidget(renderWidget_),
        slice_texture(3)
{
    transformation_matrix.identity();
    rotation_matrix.identity();
    transformation_matrix2.identity();
    rotation_matrix2.identity();
    if (cur_tracking_window.handle->has_odfs())
    {
        for (unsigned int index = 0; index < cur_tracking_window.handle->dir.odf_table.size(); ++index)
        {
            odf_color1.push_back(std::abs(cur_tracking_window.handle->dir.odf_table[index][0]));
            odf_color1.push_back(std::abs(cur_tracking_window.handle->dir.odf_table[index][1]));
            odf_color1.push_back(std::abs(cur_tracking_window.handle->dir.odf_table[index][2]));

            odf_color2.push_back(0.1f);
            odf_color2.push_back(0.1f);
            odf_color2.push_back(0.8f);

            odf_color3.push_back(0.8f);
            odf_color3.push_back(0.1f);
            odf_color3.push_back(0.1f);
        }
    }

}

void GLWidget::clean_up(void)
{
    makeCurrent();
    slice_texture.clear();
    doneCurrent();
    //tipl::out() << __FUNCTION__ << " " << __FILE__ << std::endl;
}


int GLWidget::get_param(const char* name)
{
    return renderWidget->getData(name).toInt();
}
float GLWidget::get_param_float(const char* name)
{
    return renderWidget->getData(name).toFloat();
}

bool GLWidget::check_change(const char* name,unsigned char& var)
{
    unsigned char v = uint8_t(renderWidget->getData(name).toInt());
    if(v != var)
    {
        var = v;
        return true;
    }
    return false;
}
bool GLWidget::check_change(const char* name,float& var)
{
    float v = renderWidget->getData(name).toFloat();
    if(v != var)
    {
        var = v;
        return true;
    }
    return false;
}



bool check_error(const char* line)
{
    GLenum code = glGetError();
    if(code == GL_NO_ERROR)
        return false;
    while(code)
    {
        switch(code)
        {
        case GL_INVALID_ENUM:
            tipl::out() << "GL_INVALID_ENUM at " << line << std::endl;
            break;
        case GL_INVALID_VALUE:
            tipl::out() << "GL_INVALID_VALUE at " << line << std::endl;
            break;
        case GL_INVALID_OPERATION:
            tipl::out() << "GL_INVALID_OPERATION at " << line << std::endl;
            break;
        case GL_STACK_OVERFLOW:
            tipl::out() << "GL_STACK_OVERFLOW at " << line << std::endl;
            break;
        case GL_STACK_UNDERFLOW:
            tipl::out() << "GL_STACK_UNDERFLOW at " << line << std::endl;
            break;
        case GL_OUT_OF_MEMORY:
            tipl::out() << "GL_OUT_OF_MEMORY at " << line << std::endl;
            break;
        }
    }
    return true;
}


void GLWidget::set_view(unsigned char view_option)
{
    float scale = float(std::pow(transformation_matrix.det(),1.0/3.0));
    // initialize world matrix
    transformation_matrix.identity();
    rotation_matrix.identity();


    if(get_param("scale_voxel") && cur_tracking_window.handle->vs[0] > 0.0f)
    {
        transformation_matrix[5] = cur_tracking_window.handle->vs[1] / cur_tracking_window.handle->vs[0];
        transformation_matrix[10] = cur_tracking_window.handle->vs[2] / cur_tracking_window.handle->vs[0];
    }

    tipl::vector<3,float> center_point(cur_tracking_window.handle->dim[0]/2.0-0.5,
                                        cur_tracking_window.handle->dim[1]/2.0-0.5,
                                        cur_tracking_window.handle->dim[2]/2.0-0.5);
    transformation_matrix[0] *= scale;
    transformation_matrix[5] *= scale;
    transformation_matrix[10] *= scale;
    transformation_matrix[12] = -transformation_matrix[0]*center_point[0];
    transformation_matrix[13] = -transformation_matrix[5]*center_point[1];
    transformation_matrix[14] = -transformation_matrix[10]*center_point[2];
    tipl::matrix<4,4> m;
    if(view_option != 2)
    {
        m.zero();
        m[15] = 1.0;
        switch(view_option)
        {
        case 0:
            m[2] = -1.0;
            m[4] = 1.0;
            m[9] = -1.0;
            break;
        case 1:
            m[0] = 1.0;
            m[6] = 1.0;
            m[9] = -1.0;
            break;
        case 2:
            break;
        }
        transformation_matrix *= m;
        rotation_matrix *= m;
    }
    // rotate 180 degrees
    if(set_view_flip)
    {
        m.identity();
        m[0] = -1.0;
        m[10] = -1.0;
        transformation_matrix *= m;
        rotation_matrix *= m;
    }
    transformation_matrix2 = transformation_matrix;
    rotation_matrix2 =rotation_matrix;
    set_view_flip = !set_view_flip;
}

void setupLight(float ambient,float diffuse,float specular,float angle,float angle1,unsigned char light_option)
{
    glShadeModel(GL_SMOOTH);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    if(light_option >= 1)
        glEnable(GL_LIGHT1);
    else
        glDisable(GL_LIGHT1);
    if(light_option >= 2)
        glEnable(GL_LIGHT2);
    else
        glDisable(GL_LIGHT2);
    float angle_shift = 3.1415926f*2.0f/(light_option+1.0f);

    GLfloat light[4];
    std::fill(light,light+3,diffuse);
    light[3] = 1.0f;
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light);
    glLightfv(GL_LIGHT2, GL_DIFFUSE, light);

    std::fill(light,light+3,ambient);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light);
    glLightfv(GL_LIGHT1, GL_AMBIENT, light);
    glLightfv(GL_LIGHT2, GL_AMBIENT, light);

    std::fill(light,light+3,specular);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light);
    glLightfv(GL_LIGHT1, GL_SPECULAR, light);
    glLightfv(GL_LIGHT2, GL_SPECULAR, light);


    GLfloat lightDir[4] = { 1.0f, 1.0f, -1.0f, 0.0f};
    angle1 = 3.1415926f-angle1;
    lightDir[0] = std::cos(angle)*std::sin(angle1);
    lightDir[1] = std::sin(angle)*std::sin(angle1);
    lightDir[2] = std::cos(angle1);
    glLightfv(GL_LIGHT0, GL_POSITION, lightDir);
    angle += angle_shift;
    lightDir[0] = std::cos(angle)*std::sin(angle1);
    lightDir[1] = std::sin(angle)*std::sin(angle1);
    lightDir[2] = std::cos(angle1);
    glLightfv(GL_LIGHT1, GL_POSITION, lightDir);
    angle += angle_shift;
    lightDir[0] = std::cos(angle)*std::sin(angle1);
    lightDir[1] = std::sin(angle)*std::sin(angle1);
    lightDir[2] = std::cos(angle1);
    glLightfv(GL_LIGHT2, GL_POSITION, lightDir);
    check_error(__FUNCTION__);
}

void setupMaterial(float emission,float specular,int shininess)
{
    GLfloat material2[4] = { 0.0f, 0.0f, 0.0f, 1.0f};
    std::fill(material2,material2+3,emission);
    glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,material2);

    std::fill(material2,material2+3,specular);
    glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,material2);

    glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
    check_error(__FUNCTION__);
}

unsigned char getCurView(const tipl::matrix<4,4>& m)
{
    unsigned char cur_view = 0;
    {
        const float view_dirs[6][3] = {{1,0,0},{0,1,0},{0,0,1},{-1,0,0},{0,-1,0},{0,0,-1}};
        tipl::matrix<4,4> mat = tipl::inverse(m);
        tipl::vector<3,float> dir(mat.begin()+8);
        float max_cos = 0;
        for (unsigned char index = 0;index < 6;++index)
        if (dir*tipl::vector<3,float>(view_dirs[index]) < max_cos)
        {
            max_cos = dir*tipl::vector<3,float>(view_dirs[index]);
            cur_view = index;
        }
    }
    return cur_view;
}
void handleAlpha(tipl::rgb color,
                 float alpha,int blend1,int blend2)
{
    if(alpha != 1.0f)
    {
        glEnable(GL_BLEND);
        glBlendFunc (BlendFunc1[blend1],
                     BlendFunc2[blend2]);
    }
    GLfloat material2[4] = { 0.0f, 0.0f, 0.0f, 0.5f};
    material2[0] = color.r/255.0f;
    material2[1] = color.g/255.0f;
    material2[2] = color.b/255.0f;
    material2[3] = alpha*float(color.a)/255.0f;
    glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,material2);
}

void my_gluLookAt(GLfloat eyex, GLfloat eyey, GLfloat eyez, GLfloat centerx,
          GLfloat centery, GLfloat centerz, GLfloat upx, GLfloat upy,
          GLfloat upz)
{
    tipl::vector<3,float> forward, side, up;
    GLfloat m[4][4];

    forward[0] = centerx - eyex;
    forward[1] = centery - eyey;
    forward[2] = centerz - eyez;

    up[0] = upx;
    up[1] = upy;
    up[2] = upz;

    forward.normalize();

    /* Side = forward x up */
    side = forward.cross_product(up);
    side.normalize();

    /* Recompute up as: up = side x forward */
    up = side.cross_product(forward);
    up.normalize();

    m[0][0] = side[0];
    m[1][0] = side[1];
    m[2][0] = side[2];
    m[3][0] = 0;

    m[0][1] = up[0];
    m[1][1] = up[1];
    m[2][1] = up[2];
    m[3][1] = 0;

    m[0][2] = -forward[0];
    m[1][2] = -forward[1];
    m[2][2] = -forward[2];
    m[3][2] = 0;

    m[0][3] = 0;
    m[1][3] = 0;
    m[2][3] = 0;
    m[3][3] = 1;

    glMultMatrixf(&m[0][0]);
    glTranslatef(-eyex, -eyey, -eyez);
}

void GLWidget::setFrustum(void)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double p[11] = {0.35,0.4,0.45,0.5,0.6,0.8,1.0,1.5,2.0,12.0,50.0};
    GLdouble perspective = p[get_param("perspective")];
    GLdouble zNear = 1.0;
    GLdouble zFar = 1000.0;
    GLdouble aspect = double(view_mode == view_mode_type::two ? cur_width/2:cur_width)/double(cur_height);
    GLdouble fH = 0.25;
    GLdouble fW = fH * aspect;
    glFrustum( -fW, fW, -fH, fH, zNear*perspective, zFar*perspective);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    my_gluLookAt(0.0f,0.0f,-200.0f*float(perspective),0.0f,0.0f,0.0f,0.0f,-1.0f,0.0f);
}


void GLWidget::renderText(float x,float y, const QString &str, const QFont & font)
{
    GLdouble glColor[4];
    glGetDoublev(GL_CURRENT_COLOR, glColor);
    text_pos.push_back(tipl::vector<2>(x,y));
    text_color.push_back(QColor());
    text_color.back().setRgbF(glColor[0], glColor[1], glColor[2], glColor[3]);
    text_font.push_back(font);
    text_str.push_back(str);
}
void GLWidget::renderText(float x, float y, float z, const QString &str, const QFont & font)
{
    tipl::matrix<4,4> proj,model;
    tipl::matrix<4,1> view;
    glGetFloatv(GL_MODELVIEW_MATRIX,model.begin());
    glGetFloatv(GL_PROJECTION_MATRIX, proj.begin());
    glGetFloatv(GL_VIEWPORT, view.begin());
    tipl::matrix<1,4> in = {x,y,z,1.0f};
    in = tipl::matrix<1,4>(in*model)*proj;
    if (in[3] == 0.0f)
        return;
    renderText(view[0] + (1 + in[0]/in[3]) * view[2] * 0.5f,
               cur_height - (view[1] + (1 + in[1]/in[3]) * view[3] * 0.5f),
               str,font);
}

void GLWidget::initializeGL()
{
    tipl::progress prog("initializing OpenGL");
    initializeOpenGLFunctions();
    if(!isValid() || !glGetString(GL_VERSION))
    {
        QMessageBox::critical(this,"ERROR","System has no OpenGL support. 3D visualization is disabled. Please update or install graphic card driver.");
        return;
    }
    tipl::out() << "openGL information" << std::endl;
    tipl::out() << "version: " << glGetString(GL_VERSION) << std::endl;
    tipl::out() << "vendor: " << glGetString(GL_VENDOR) << std::endl;
    tipl::out() << "renderer: " << glGetString(GL_RENDERER) << std::endl;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
    glBlendFunc (GL_DST_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    scale_by(0.5);
}
void GLWidget::paintGL()
{
    if(no_update)
        return;
    if(check_error(__FUNCTION__))
    {
        no_update = true;
        return;
    }

    QPainter painter(this);
    painter.beginNativePainting();
    glEnable(GL_DEPTH_TEST);


    int color = get_param("bkg_color");
    glClearColor(float((color & 0x00FF0000) >> 16)/255.0f,
                  float((color & 0x0000FF00) >> 8)/255.0f,
                  float(color & 0x000000FF)/255.0f,1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    setFrustum();
    if(check_error("basic"))
        return;
    {
        if(check_change("scale_voxel",scale_voxel))
        {
            set_view(2);// initialize view to axial
            if(get_param("orientation_convention") == 1)
                set_view(2);
        }

        if(get_param("anti_aliasing"))
            glEnable(GL_MULTISAMPLE);
        else
            glDisable(GL_MULTISAMPLE);
        glGetError(); //silence the multisample error

        if(get_param("line_smooth"))
        {
            glEnable(GL_LINE_SMOOTH);
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        }
        else
            glDisable(GL_LINE_SMOOTH);

        if(get_param("point_smooth"))
        {
            glEnable(GL_POINT_SMOOTH);
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
        }
        else
            glDisable(GL_POINT_SMOOTH);

        if(get_param("poly_smooth"))
        {
            glEnable(GL_POLYGON_SMOOTH);
            glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
        }
        else
            glDisable(GL_POLYGON_SMOOTH);
    }

    check_error("others");

    switch(view_mode)
    {
        case view_mode_type::single:
            glViewport(0,0, cur_width, cur_height);
            //if(!get_param("stereoscopy"))
            renderLR();
            /*
            else
            {
                glDrawBuffer(GL_BACK_RIGHT);
                renderLR();
                glDrawBuffer(GL_BACK_LEFT);
                renderLR();
                glDrawBuffer(GL_BACK);
            }
            */
            break;
        case view_mode_type::two:
            glViewport(0,0, cur_width/2, cur_height);
            renderLR();
            transformation_matrix.swap(transformation_matrix2);
            rotation_matrix.swap(rotation_matrix2);
            glViewport(cur_width/2,0, cur_width/2, cur_height);
            renderLR();
            transformation_matrix.swap(transformation_matrix2);
            rotation_matrix.swap(rotation_matrix2);
            break;
        case view_mode_type::stereo:
            glViewport(0,0, cur_width/2, cur_height);
            renderLR();
            tipl::matrix<4,4> T(transformation_matrix);
            // add a rotation to the transformation matrix
            glPushMatrix();
            glLoadIdentity();
            glRotated(get_param_float("stereoscopy_angle"),0,1,0);
            glMultMatrixf(transformation_matrix.begin());
            glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix.begin());
            glPopMatrix();

            glViewport(cur_width/2,0, cur_width/2, cur_height);
            renderLR();
            transformation_matrix2 = transformation_matrix;
            rotation_matrix2 = rotation_matrix;
            transformation_matrix = T;
            break;

    }

    painter.endNativePainting();
    painter.end();

    // draw text here
    {
        QPainter text_painter(this);
        for(size_t i = 0;i < text_str.size();++i)
        {
            text_painter.setPen(text_color[i]);
            text_painter.setFont(text_font[i]);
            text_painter.drawText(text_pos[i].x(), text_pos[i].y(), text_str[i]);
        }
        text_painter.end();
        text_pos.clear();
        text_color.clear();
        text_font.clear();
        text_str.clear();
    }
    // provide 3D view to ROI window
    if(cur_tracking_window["roi_layout"].toInt() == 1)// 3 slice
    {
        if(!video_capturing)
        {
            video_capturing = true;
            cur_tracking_window.scene.update_3d(grabFramebuffer());
            video_capturing = false;
        }
    }
    // handle capture video
    if(!video_capturing && video_handle.get())
    {
        video_capturing = true;
        QBuffer buffer;
        QImageWriter writer(&buffer, "JPG");
        QImage I = grabFramebuffer();
        writer.write(I);
        QByteArray data = buffer.data();
        video_handle->add_frame(data.begin(),uint32_t(data.size()),true);
        video_capturing = false;
        if(video_frames > 10000)
            record_video();
    }
}

void CylinderGL(GLUquadricObj* ptr,const tipl::vector<3>& p1,const tipl::vector<3>& p2,double r)
{
     tipl::vector<3> z(0,0,1),dis(p1-p2);
     tipl::vector<3> t = z.cross_product(dis);
     double v = dis.length();
     float angle = 180.0f / 3.1415926f*float(std::acos(double(dis[2])/v));
     glPushMatrix();
     glTranslatef(p2[0],p2[1],p2[2]);
     glRotatef(angle,t[0],t[1],t[2]);
     gluCylinder(ptr,r,r,v,10,int(v/10)+1);		// Draw A cylinder
     glPopMatrix();
}

void GLWidget::renderLR()
{
    std::shared_ptr<SliceModel> current_slice = cur_tracking_window.current_slice;
    if (cur_tracking_window.handle->has_odfs() &&
        get_param("show_odf"))
    {
        float fa_threshold = cur_tracking_window.get_fa_threshold();
        if(check_param("odf_position",odf_position)+
           check_param("odf_skip",odf_skip)+
           check_param("odf_shape",odf_shape)+
           check_param("odf_scale",odf_scale)+
           check_param("odf_color",odf_color)+
           check_param("odf_position",odf_position) > 0 ||
           odf_dim != cur_tracking_window.cur_dim || odf_slide_pos != current_slice->slice_pos[cur_tracking_window.cur_dim])
        {
            odf_dim = cur_tracking_window.cur_dim;
            odf_slide_pos = current_slice->slice_pos[cur_tracking_window.cur_dim];
            odf_points.clear();
        }

        std::shared_ptr<fib_data> handle = cur_tracking_window.handle;
        unsigned char skip_mask_set[3] = {0,1,3};
        unsigned char mask = skip_mask_set[odf_skip];
        std::shared_ptr<SliceModel> slice =
                (cur_tracking_window.current_slice->is_diffusion_space?
                     cur_tracking_window.current_slice:
                     cur_tracking_window.slices[0]);

        auto& geo = cur_tracking_window.handle->dim;
        if(odf_points.empty())
        {
            std::vector<tipl::pixel_index<3> > odf_pos;
            switch(odf_position) // along slide
            {

            case 0:
                {
                    tipl::shape<2> geo2(slice->dim[odf_dim==0?1:0],
                                           slice->dim[odf_dim==2?1:2]);
                    for(tipl::pixel_index<2> index(geo2);index < geo2.size();++index)
                    {
                        if((index[0] & mask) | (index[1] & mask))
                            continue;
                        auto xyz = slice->to3DSpace<tipl::vector<3,int> >(cur_tracking_window.cur_dim,index[0],index[1]);
                        if (!slice->dim.is_valid(xyz))
                            continue;
                        tipl::pixel_index<3> pos(xyz.begin(),geo);
                        if (handle->dir.fa[0][pos.index()] <= fa_threshold)
                            continue;
                        odf_pos.push_back(pos);
                    }
                }
                break;
            case 1: // intersection
                odf_pos.push_back(tipl::pixel_index<3>(slice->slice_pos[0],slice->slice_pos[1],slice->slice_pos[2],
                                              geo));
                break;
            case 2: //all
                for(tipl::pixel_index<3> index(geo);index < geo.size();++index)
                {
                    if(((index[0] & mask) | (index[1] & mask) | (index[2] & mask)) ||
                       handle->dir.fa[0][index.index()] <= fa_threshold)
                        continue;
                    odf_pos.push_back(index);
                }
                break;
            }
            add_odf(odf_pos);
        }

        float* odf_color_ptr = (float*)&odf_color1.front();
        switch(odf_color)
        {
            case 1:
                odf_color_ptr = (float*)&odf_color2.front();
                break;
            case 2:
                odf_color_ptr = (float*)&odf_color3.front();
                break;
        }
        glEnable(GL_COLOR_MATERIAL);
        setupLight(float(get_param("odf_light_ambient"))/10.0f,
                   float(get_param("odf_light_diffuse"))/10.0f,
                   float(get_param("odf_light_specular"))/10.0f,
                   float(get_param("odf_light_dir"))*3.1415926f*2.0f/10.0f,
                   float(get_param("odf_light_shading"))*3.1415926f/20.0f,
                   get_param("odf_light_option"));
        setupMaterial(float(get_param("odf_emission"))/10.0f,
                      float(get_param("odf_specular"))/10.0f,
                      get_param("odf_shininess")*10);

        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        unsigned int num_odf = odf_points.size()/cur_tracking_window.handle->dir.odf_table.size();
        unsigned int face_size = cur_tracking_window.handle->dir.odf_faces.size()*3;


        for(unsigned int index = 0,base_index = 0;index < num_odf;
            ++index,base_index += cur_tracking_window.handle->dir.odf_table.size())
        {
            glVertexPointer(3, GL_FLOAT, 0, (float*)&odf_points[base_index]);
            glColorPointer(3, GL_FLOAT, 0, odf_color_ptr);
            glNormalPointer(GL_FLOAT, 0, (float*)&odf_norm[base_index]);
            glDrawElements(GL_TRIANGLES, face_size,
                           GL_UNSIGNED_SHORT,handle->dir.odf_faces[0].begin());
        }
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
        glPopMatrix();
        glDisable(GL_COLOR_MATERIAL);





    }

    if (cur_tracking_window.tractWidget->tract_rendering.size() && get_param("show_tract"))
    {
        glLineWidth (1.0F);
        glEnable(GL_COLOR_MATERIAL);
        if(get_param("tract_style") != 1 ||  // 1 = tube
           get_param("tract_light_option") == 2)
        {
            glDisable(GL_LIGHTING);
            glLineWidth(get_param("tract_line_width"));
        }
        else
            setupLight(float(get_param("tract_light_ambient"))/10.0f,
                   float(get_param("tract_light_diffuse"))/10.0f,
                   float(get_param("tract_light_specular"))/10.0f,
                   float(get_param("tract_light_dir"))*3.1415926f*2.0f/10.0f,
                   float(get_param("tract_light_shading"))*3.1415926f/20.0f,
                   get_param("tract_light_option"));


        if(get_param("tract_shader") && get_param("tract_light_option"))
        {
            glEnable(GL_LIGHT2);
            GLfloat light[4];
            std::fill(light,light+3, float(get_param("tract_light_diffuse"))/10.0f);
            light[3] = 1.0f;
            glLightfv(GL_LIGHT2, GL_DIFFUSE, light);
            std::fill(light,light+3,0.0f);
            glLightfv(GL_LIGHT2, GL_AMBIENT, light);
            std::fill(light,light+3,0.0f);
            glLightfv(GL_LIGHT2, GL_SPECULAR, light);
            GLfloat lightDir[4] = { 0.0f, 0.0f, -1.0f, 0.0f};
            glLightfv(GL_LIGHT2, GL_POSITION, lightDir);
        }

        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());
        setupMaterial((float)(get_param("tract_emission"))/10.0,
                      (float)(get_param("tract_specular"))/10.0,
                      get_param("tract_shininess")*10);



        if(get_param("tract_color_style") != tract_color_style &&
           cur_tracking_window.color_bar.get())
        {
            if(get_param("tract_color_style") > 1)
                cur_tracking_window.color_bar->show();
            else
                cur_tracking_window.color_bar->hide();
        }


        {
            bool changed = false;
            while(
               check_change("tract_alpha",tract_alpha) ||
               check_change("tract_style",tract_style) ||
               check_change("tract_color_style",tract_color_style) ||
               check_change("tract_color_saturation",tract_color_saturation) ||
               check_change("tract_color_brightness",tract_color_brightness) ||
               check_change("tube_diameter",tube_diameter) ||
               check_change("tract_tube_detail",tract_tube_detail) ||
               check_change("tract_shader",tract_shader) ||
               check_change("end_point_shift",end_point_shift))
                changed = true;
            if(changed)
                cur_tracking_window.tractWidget->need_update_all();
        }

        glDepthMask(true);
        if(get_param_float("tract_alpha") != 1.0)
        {
            glEnable(GL_BLEND);
            glBlendFunc (BlendFunc1[get_param("tract_bend1")],
                         BlendFunc2[get_param("tract_bend2")]);
        }
        else
            glDisable(GL_BLEND);

        cur_tracking_window.tractWidget->render_tracts(this);

        glPopMatrix();
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);
        glDepthMask(true);
        check_error("show_tract");
    }

    if (get_param("show_slice"))
    {
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);
        float alpha = get_param_float("slice_alpha");
        handleAlpha(tipl::rgb(0,0,0,255),
                        alpha,get_param("slice_bend1"),get_param("slice_bend2"));
        glDepthMask((alpha == 1.0));

        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());


        std::vector<tipl::vector<3,float> > points(4);

        bool changed_slice = check_change("slice_match_bkcolor",slice_match_bkcolor);
        for(unsigned int dim = 0;dim < slice_texture.size();++dim)
        {
            if((dim == 0 && !cur_tracking_window.ui->glSagCheck->checkState()) ||
               (dim == 1 && !cur_tracking_window.ui->glCorCheck->checkState()) ||
               (dim == 2 && !cur_tracking_window.ui->glAxiCheck->checkState()))
                continue;

            if(dim < 3 && (slice_pos[dim] != current_slice->slice_pos[dim] || changed_slice))
            {
                tipl::color_image texture;
                if(current_slice->handle && current_slice->handle->has_high_reso)
                    current_slice->get_high_reso_slice(texture,dim,current_slice->slice_pos[dim]);
                else
                    current_slice->get_slice(texture,dim,current_slice->slice_pos[dim],cur_tracking_window.overlay_slices);

                if(get_param("slice_match_bkcolor"))
                {
                    auto slice_bk = texture[0];
                    uint32_t bkcolor = get_param("bkg_color");
                    for(size_t index = 0;index < texture.size();++index)
                        if(texture[index] == slice_bk)
                            texture[index] = bkcolor;
                }

                for(unsigned int index = 0;index < texture.size();++index)
                {
                    unsigned char value =
                    255-texture[index].data[0];
                    if(value >= 230)
                        value -= (value-230)*10;
                    texture[index].data[3] = value;
                }

                slice_texture[dim] = std::make_shared<QOpenGLTexture>((QImage() << texture).mirrored());
                slice_pos[dim] = current_slice->slice_pos[dim];
            }

            if(slice_texture[dim].get())
            {
                slice_texture[dim]->bind();
                int texparam[] = {GL_NEAREST,
                                   GL_LINEAR};
                glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,texparam[get_param("slice_mag_filter")]);

                glBegin(GL_QUADS);
                glColor4f(1.0,1.0,1.0,std::min(alpha+0.2,1.0));

                if(dim < 3)
                    slice_location(dim,points);
                else
                    points = keep_slice_points;
                glTexCoord2f(0.0f, 1.0f);
                glVertex3f(points[0][0],points[0][1],points[0][2]);
                glTexCoord2f(1.0f, 1.0f);
                glVertex3f(points[1][0],points[1][1],points[1][2]);
                glTexCoord2f(1.0f, 0.0f);
                glVertex3f(points[3][0],points[3][1],points[3][2]);
                glTexCoord2f(0.0f, 0.0f);
                glVertex3f(points[2][0],points[2][1],points[2][2]);
                glEnd();
            }
            if(keep_slice && dim == cur_tracking_window.cur_dim)
            {
                if(slice_texture.size() > 3)
                    slice_texture.pop_back();
                slice_texture.push_back(slice_texture[dim]);
                slice_texture[dim].reset();
                keep_slice_points = points;
                keep_slice = false;
            }
        }

        glPopMatrix();
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);
        glDepthMask(true);
        check_error("show_slice");
    }
    if (!cur_tracking_window.deviceWidget->devices.empty() &&
        get_param("show_device"))
    {
        glDisable(GL_COLOR_MATERIAL);
        setupLight(float(get_param("device_light_ambient"))/10.0f,
                   float(get_param("device_light_diffuse"))/10.0f,
                   float(get_param("device_light_specular"))/10.0f,
                   float(get_param("device_light_dir"))*3.1415926f*2.0f/10.0f,
                   float(get_param("device_light_shading"))*3.1415926f/20.0f,
                   uint8_t(get_param("device_light_option")));

        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());
        setupMaterial(float(get_param("device_emission"))/10.0f,
                      float(get_param("device_specular"))/10.0f,
                      get_param("device_shininess")*10);

        int blend1 = get_param("device_bend1");
        int blend2 = get_param("device_bend2");
        glEnable(GL_BLEND);
        glBlendFunc (BlendFunc1[blend1],
                     BlendFunc2[blend2]);

        glEnable(GL_COLOR_MATERIAL);
        if(!DeviceQua.get())
        {
            DeviceQua.reset(new GluQua);
            gluQuadricNormals(DeviceQua->get(), GLU_SMOOTH);
        }

        float vs = 1.0f/cur_tracking_window.handle->vs[0];
        auto& devices = cur_tracking_window.deviceWidget->devices;
        for(size_t index = 0;index < devices.size();++index)
        if(cur_tracking_window.deviceWidget->item(int(index),0)->checkState() == Qt::Checked)
        {
            glPushMatrix();
            // oriented and move the device
            glTranslatef(devices[index]->pos[0],devices[index]->pos[1],devices[index]->pos[2]);
            tipl::vector<3> z(0,0,1);
            tipl::vector<3> t = z.cross_product(devices[index]->dir);
            float angle = 180.0f / 3.1415926f*float(std::acos(double(devices[index]->dir[2])));
            glRotatef(angle,t[0],t[1],t[2]);
            glScalef(vs,vs,vs);
            std::vector<float> seg_length;
            std::vector<char> seg_type;
            float radius = 0.0f;
            devices[index]->get_rendering(seg_length,seg_type,radius);
            if(device_selected && index == selected_index)
                radius += 0.2f;
            float r = devices[index]->color.r/255.0f;
            float g = devices[index]->color.g/255.0f;
            float b = devices[index]->color.b/255.0f;
            float a = devices[index]->color.a/255.0f;

            auto Cylinder = [&](float r,float length,int s = 20)
                {gluCylinder(DeviceQua->get(),double(r),double(r),double(length),s,std::max<int>(1,int(length)));};
            // head
            for(size_t j = 0;j < seg_length.size();++j)
            {
                switch(seg_type[j])
                {
                    case -2: // cylindercap
                        glColor4f(r,g,b,a);
                        gluDisk(DeviceQua->get(),radius,radius+5,8,2);
                        Cylinder(radius+5,seg_length[j],8);
                        Cylinder(radius,seg_length[j],8);
                        glPushMatrix();
                        glTranslatef(0.0f,0.0f,seg_length[j]);
                        gluDisk(DeviceQua->get(),radius,radius+5,8,2);
                        glPopMatrix();
                        break;
                    case -1: // tip ball
                        if(j+1 < seg_type.size() && seg_type[j+1] == 0)
                            glColor4f(r,g,b,a);
                        else
                            glColor4f(0.2f,0.2f,0.2f,a);
                        glTranslatef(0.0f,0.0f,radius*0.5f);
                        seg_length[j+1] -= radius*0.5f;
                        gluSphere(DeviceQua->get(),double(radius),20,10);
                        continue;
                    case 0:
                    case 3:
                        glColor4f(r,g,b,a);
                        Cylinder(radius,seg_length[j]);
                        break;
                    case 1:
                        glColor4f(0.2f,0.2f,0.2f,a);
                        Cylinder(radius,seg_length[j]);
                        break;
                    case 2:
                        glColor4f(r,g,b,a);
                        Cylinder(radius,seg_length[j]);
                        glPushMatrix();
                        glTranslatef(radius*0.2f,0.0f,0.0f);
                        glColor4f(0.2f,0.2f,0.2f,a);
                        Cylinder(radius*0.9f,seg_length[j]);
                        glPopMatrix();
                        glPushMatrix();
                        glTranslatef(-radius*0.1414f,radius*0.1414f,0.0f);
                        Cylinder(radius*0.9f,seg_length[j]);
                        glPopMatrix();
                        glPushMatrix();
                        glTranslatef(-radius*0.1414f,-radius*0.1414f,0.0f);
                        Cylinder(radius*0.9f,seg_length[j]);
                        glPopMatrix();
                        break;
                }
                glTranslatef(0.0f,0.0f,seg_length[j]);
            }
            glPopMatrix();
        }


        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_BLEND);
        glPopMatrix();
        check_error("show_device");
    }

    std::vector<bool> region_visualized(cur_tracking_window.regionWidget->regions.size());
    if (get_param("show_region"))
    {
        glDisable(GL_COLOR_MATERIAL);
        setupLight((float)(get_param("region_light_ambient"))/10.0,
                   (float)(get_param("region_light_diffuse"))/10.0,
                   (float)(get_param("region_light_specular"))/10.0,
                   (float)(get_param("region_light_dir"))*3.1415926*2.0/10.0,
                   (float)(get_param("region_light_shading"))*3.1415926/20.0,
                   get_param("region_light_option"));
        setupMaterial((float)(get_param("region_emission"))/10.0,
                      (float)(get_param("region_specular"))/10.0,
                      get_param("region_shininess")*10);
        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());



        if(get_param("region_graph") && !connectivity.empty())
        {
            tipl::value_to_color<float> v2c;
            v2c.two_color((unsigned int)get_param("region_edge_color2"),
                          (unsigned int)get_param("region_edge_color1"));
            v2c.set_range(get_param_float("region_edge_value2"),
                          get_param_float("region_edge_value1"));
            if (cur_tracking_window.regionWidget->regions.size())
            {
                glEnable(GL_COLOR_MATERIAL);
                if(!RegionSpheres.get())
                {
                    RegionSpheres.reset(new GluQua);
                    gluQuadricNormals(RegionSpheres->get(), GLU_SMOOTH);
                }
                float edge_threshold = get_param_float("region_edge_threshold")*max_connectivity;
                if(!connectivity.empty() && connectivity.width() == int(cur_tracking_window.regionWidget->regions.size()) && max_connectivity != 0.0f)
                {
                    for(unsigned int i = 0;i < cur_tracking_window.regionWidget->regions.size();++i)
                        for(unsigned int j = i+1;j < cur_tracking_window.regionWidget->regions.size();++j)
                        if(cur_tracking_window.regionWidget->item(int(i),0)->checkState() == Qt::Checked &&
                           cur_tracking_window.regionWidget->item(int(j),0)->checkState() == Qt::Checked &&
                           !cur_tracking_window.regionWidget->regions[i]->region.empty() &&
                                !cur_tracking_window.regionWidget->regions[j]->region.empty())
                        {
                            float c = std::fabs(connectivity.at(i,j));
                            if(c == 0.0f || (edge_threshold != 0.0f && c < edge_threshold))
                                continue;
                            region_visualized[i] = true;
                            region_visualized[j] = true;
                            auto centeri = cur_tracking_window.regionWidget->regions[i]->get_center();
                            auto centerj = cur_tracking_window.regionWidget->regions[j]->get_center();
                            auto color = v2c[connectivity.at(i,j)/max_connectivity];
                            glColor4f(((float)color.r)/255.0f,
                                      ((float)color.g)/255.0f,
                                      ((float)color.b)/255.0f,1.0f);
                            CylinderGL(RegionSpheres->get(),centeri,centerj,
                                       double((get_param("region_constant_edge_size") ? 0.5f:c/max_connectivity)*(get_param("region_edge_size")+5)/5.0f));
                        }
                }


                for(unsigned int i = 0;i < cur_tracking_window.regionWidget->regions.size();++i)
                    if(cur_tracking_window.regionWidget->item(i,0)->checkState() == Qt::Checked &&
                       !cur_tracking_window.regionWidget->regions[i]->region.empty())
                    {
                        if(get_param("region_hide_unconnected_node") && !region_visualized[i])
                            continue;
                        auto& cur_region = cur_tracking_window.regionWidget->regions[i];
                        auto center = cur_tracking_window.regionWidget->regions[i]->get_center();
                        glPushMatrix();
                        glTranslatef(center[0],center[1],center[2]);
                        glColor4f(cur_region->region_render.color.r/255.0f,
                                  cur_region->region_render.color.g/255.0f,
                                  cur_region->region_render.color.b/255.0f,1.0f);
                        gluSphere(RegionSpheres->get(),
                                  std::pow(cur_tracking_window.regionWidget->regions[get_param("region_constant_node_size") ? 0:i]->region.size(),1.0f/3.0f)
                                  *(get_param("region_node_size")+5)/50.0f,10,10);
                        glPopMatrix();
                    }

                glDisable(GL_COLOR_MATERIAL);
            }
        }
        else
        {
            float alpha = get_param_float("region_alpha");
            unsigned char cur_view = (alpha == 1.0f ? 0 : getCurView(transformation_matrix));
            auto& regionWidget = cur_tracking_window.regionWidget;
            auto& regions = regionWidget->regions;
            std::vector<unsigned int> region_need_update;
            for(unsigned int index = 0;index < regions.size();++index)
                if(regionWidget->item(int(index),0)->checkState() == Qt::Checked &&
                        regions[index]->modified)
                    region_need_update.push_back(index);

            unsigned char smoothed = get_param("region_mesh_smoothed");
            tipl::par_for(region_need_update.size(),[&](unsigned int index){
                regions[region_need_update[index]]->makeMeshes(smoothed);
            });

            for(unsigned int index = 0;index < regions.size();++index)
                if(regionWidget->item(int(index),0)->checkState() == Qt::Checked &&
                   !regions[index]->region.empty())
                {
                    regions[index]->region_render.draw(
                               cur_view,alpha,get_param("region_bend1"),get_param("region_bend2"));
                    region_visualized[index] = true;
                }
        }
        glDisable(GL_BLEND);
        glPopMatrix();
        check_error("show_region");

    }

    if (surface.get() && get_param("show_surface"))
    {
        glDisable(GL_COLOR_MATERIAL);
        setupLight(float(get_param("surface_light_ambient"))/10.0f,
                   float(get_param("surface_light_diffuse"))/10.0f,
                   float(get_param("surface_light_specular"))/10.0f,
                   float(get_param("surface_light_dir"))*3.1415926f*2.0f/10.0f,
                   float(get_param("surface_light_shading"))*3.1415926f/20.0f,
                   get_param("surface_light_option"));

        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());
        setupMaterial(float(get_param("surface_emission"))/10.0f,
                      float(get_param("surface_specular"))/10.0f,
                      get_param("surface_shininess")*10);

        float alpha = get_param_float("surface_alpha");
        surface->color = (unsigned int)get_param("surface_color");
        surface->color.a = 255;
        surface->draw((alpha == 1.0 ? 0 : getCurView(transformation_matrix)),
                      alpha,get_param("surface_bend1"),get_param("surface_bend2"));
        glDisable(GL_BLEND);
        glPopMatrix();
        check_error("show_surface");
    }
    if (get_param("show_label"))
    {
        glEnable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());
        if (get_param("show_track_label"))
        {
            int color = get_param("track_label_color");
            int font_size = get_param("track_label_size");
            glColor3ub((color & 0x00FF0000) >> 16,(color & 0x0000FF00) >> 8,color & 0x000000FF);
            QFont font;
            font.setPointSize(font_size);
            font.setBold(get_param("track_label_bold"));
            for (int i = 0;i < cur_tracking_window.tractWidget->rowCount();++i)
            if(cur_tracking_window.tractWidget->item(i,0)->checkState() == Qt::Checked)
            {
                auto active_tract_model = cur_tracking_window.tractWidget->tract_models[size_t(i)];
                if (active_tract_model->get_visible_track_count() == 0)
                    continue;
                if(get_param("show_track_label_location"))
                {
                    renderText(cur_width/2-font_size*cur_tracking_window.tractWidget->item(i,0)->text().size()/3.5,
                               get_param("show_track_label_location") == 2 ? height()-font_size : height()/2,
                            cur_tracking_window.tractWidget->item(i,0)->text(),font);
                }
                else
                {
                    const auto& t = active_tract_model->get_tract(0);
                    size_t pos = t.size()/6*3;
                    renderText(double(t[pos]),double(t[pos+1]),double(t[pos+2]),
                            cur_tracking_window.tractWidget->item(i,0)->text(),font);
                }
            }
        }
        if (get_param("show_region_label"))
        {
            int color = get_param("region_label_color");
            glColor3ub((color & 0x00FF0000) >> 16,(color & 0x0000FF00) >> 8,color & 0x000000FF);
            QFont font;
            font.setPointSize(get_param("region_label_size"));
            font.setBold(get_param("region_label_bold"));
            auto& regions = cur_tracking_window.regionWidget->regions;
            for(unsigned int i = 0;i < regions.size();++i)
                if(region_visualized[i])
                {
                    auto* item = cur_tracking_window.regionWidget->item(int(i),0);
                    auto p = regions[i]->get_center();
                    if(p[0] == 0.0f && p[1] == 0.0f && p[2] == 0.0f)
                    {
                        const auto& p2 = regions[i]->region_render.center;
                        tipl::vector<3> p3(p2);
                        renderText(p3[0],p3[1],p3[2],item->text(),font);
                    }
                    else
                        renderText(p[0],p[1],p[2],item->text(),font);
                }
        }
        if (get_param("show_device_label"))
        {
            int color = get_param("device_label_color");
            glColor3ub((color & 0x00FF0000) >> 16,(color & 0x0000FF00) >> 8,color & 0x000000FF);
            QFont font;
            font.setPointSize(get_param("device_label_size"));
            font.setBold(get_param("device_label_bold"));
            auto& devices = cur_tracking_window.deviceWidget->devices;
            for(unsigned int i = 0;i < devices.size();++i)
                if(cur_tracking_window.deviceWidget->item(int(i),0)->checkState() == Qt::Checked)
                {
                    auto p = devices[i]->pos + devices[i]->dir*(devices[i]->length/cur_tracking_window.handle->vs[0]);
                    renderText(p[0],p[1],p[2],cur_tracking_window.deviceWidget->item(int(i),0)->text(),font);
                }
        }
        glPopMatrix();
        glEnable(GL_LIGHTING);
        glEnable(GL_DEPTH_TEST);

    }

    if (get_param("show_directional_axis"))
    {
        float L2 = get_param_float("axis_line_length")*0.75f;

        glEnable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        float p[11] = {0.35f,0.4f,0.45f,0.5f,0.6f,0.8f,1.0f,1.5f,2.0f,12.0f,50.0f};
        GLfloat perspective = p[get_param("perspective")];
        GLfloat zNear = 1.0f;
        GLfloat zFar = 1000.0f;
        GLfloat aspect = float(view_mode == view_mode_type::two ? cur_width/2:cur_width)/float(cur_height);
        GLfloat fH = 0.25f;
        GLfloat fW = fH * aspect;
        glFrustum( -fW, std::sqrt(L2)*0.015f,std::sqrt(L2)*-0.015f, fH, zNear*perspective, zFar*perspective);


        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        my_gluLookAt(0,0,-200.0f*perspective,0,0,0,0,-1.0f,0);
        glMultMatrixf(rotation_matrix.begin());
        glLineWidth (get_param_float("axis_line_thickness"));
        glBegin (GL_LINES);
        glColor3f (0.8f,0.5f,0.5f);  glVertex3f(0,0,0);  glVertex3f(L2,0,0);    // X axis is red.
        glColor3f (0.5f,0.8f,0.5f);  glVertex3f(0,0,0);  glVertex3f(0,L2,0);    // Y axis is green.
        glColor3f (0.5f,0.5f,0.8f);  glVertex3f(0,0,0);  glVertex3f(0,0,L2);    // z axis is blue.
        glEnd();
        if(get_param("show_axis_label"))
        {
            QFont font;
            font.setPointSize(get_param("axis_label_size"));
            font.setBold(get_param("axis_label_bold"));
            glColor3f (0.3f,0.3f,1.0f);
            renderText(0,0,L2,"S",font);
            glColor3f (1.0f,0.3f,0.3f);
            renderText(L2,0,0,"L",font);
            glColor3f (0.3f,1.0f,0.3f);
            renderText(0,L2,0,"P",font);
        }
        // This keep shade on other regions black
        glColor3f (0.0f,0.0f,0.0f);
        glEnable(GL_DEPTH_TEST);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        check_error("axis");
    }

    if(editing_option == selecting)
    {
        glEnable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        {
            glLoadIdentity();
            glOrtho(0,cur_width,cur_height,0,0.0,100.0);

            glDisable(GL_DEPTH_TEST);

            glLineWidth(4);
            glBegin (GL_LINES);
            glColor3f (0.5f,0.2f,0.2f);
            glVertex3f(lastPos.x()*devicePixelRatio(),lastPos.y()*devicePixelRatio(),0);
            glVertex3f(curPos.x()*devicePixelRatio(),curPos.y()*devicePixelRatio(),0);
            glEnd();

            glLineWidth(2);
            glBegin (GL_LINES);
            glColor3f (0.8f,0.5f,0.5f);
            glVertex3f(lastPos.x()*devicePixelRatio(),lastPos.y()*devicePixelRatio(),0);
            glVertex3f(curPos.x()*devicePixelRatio(),curPos.y()*devicePixelRatio(),0);
            glEnd();

        // This keep shade on other regions black
            glColor3f (0.0f,0.0f,0.0f);
            glEnable(GL_DEPTH_TEST);
        }
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        check_error("selection");
    }

}


void GLWidget::add_odf(const std::vector<tipl::pixel_index<3> >& odf_pos_)
{
    if(!odf.get())
        odf.reset(new odf_data);
    if(!odf->read(cur_tracking_window.handle->mat_reader))
    {
        if(!odf->error_msg.empty())
            QMessageBox::critical(this,"ERROR",odf->error_msg.c_str());
        return;
    }
    std::shared_ptr<fib_data> handle = cur_tracking_window.handle;
    std::vector<const float*> odf_buffers;
    std::vector<tipl::pixel_index<3> > odf_pos;
    for(size_t i = 0;i < odf_pos_.size();++i)
    {
        const float* odf_buffer = odf->get_odf_data(uint32_t(odf_pos_[i].index()));
        if(!odf_buffer)
            continue;
        odf_buffers.push_back(odf_buffer);
        odf_pos.push_back(odf_pos_[i]);
    }

    unsigned int odf_dim = uint32_t(cur_tracking_window.handle->dir.odf_table.size());
    unsigned int half_odf = odf_dim >> 1;
    odf_points.resize(odf_pos.size()*odf_dim);
    odf_norm.resize(odf_pos.size()*odf_dim);
    bool odf_min_max = get_param("odf_min_max");
    bool odf_smoothing = get_param("odf_smoothing");
    unsigned int odf_shape = get_param("odf_shape");
    ODFShaping shaping;
    tessellated_icosahedron ti;
    shaping.init(ti);

    tipl::par_for(odf_pos.size(),[&](size_t i)
    {

        const float* odf_buffer = odf_buffers[i];

        float odf_min = tipl::min_value(odf_buffer,odf_buffer+half_odf);
        if(odf_min < 0 || !odf_min_max)
            odf_min = 0;
        // smooth the odf a bit

        std::vector<float> new_odf_buffer;
        if(odf_smoothing)
        {
            new_odf_buffer.resize(half_odf);
            std::copy(odf_buffer,odf_buffer+half_odf,new_odf_buffer.begin());
            auto& odf_faces = handle->dir.odf_faces;
            for(size_t index = 0;index < odf_faces.size();++index)
            {
                unsigned short f1 = odf_faces[index][0];
                unsigned short f2 = odf_faces[index][1];
                unsigned short f3 = odf_faces[index][2];
                if(f1 >= half_odf)
                    f1 -= half_odf;
                if(f2 >= half_odf)
                    f2 -= half_odf;
                if(f3 >= half_odf)
                    f3 -= half_odf;
                float sum = odf_buffer[f1]+odf_buffer[f2]+odf_buffer[f3]-odf_min-odf_min-odf_min;
                sum *= 0.1f;
                sum += odf_min;
                if(odf_buffer[f1] == odf_min)
                    new_odf_buffer[f1] = std::max(sum,new_odf_buffer[f1]);
                if(odf_buffer[f2] == odf_min)
                    new_odf_buffer[f2] = std::max(sum,new_odf_buffer[f2]);
                if(odf_buffer[f3] == odf_min)
                    new_odf_buffer[f3] = std::max(sum,new_odf_buffer[f3]);
            }
            odf_buffer = &new_odf_buffer[0];
        }
        if(odf_shape)
        {
            std::vector<float> odf(half_odf);
            std::copy(odf_buffer,odf_buffer+half_odf,odf.begin());

            shaping.shape(odf,uint16_t(std::max_element(odf.begin(),odf.end())-odf.begin()));

            if(odf_shape == 1)
            {
                new_odf_buffer.resize(half_odf);
                std::copy(odf_buffer,odf_buffer+half_odf,new_odf_buffer.begin());
                tipl::minus(new_odf_buffer,odf);
            }
            else
            {
                new_odf_buffer.swap(odf);
                odf_min = 0;
            }
            odf_buffer = &new_odf_buffer[0];

        }
        auto iter = odf_points.begin()+odf_dim*i;
        std::fill(iter,iter + odf_dim,odf_pos[i]);

        for(unsigned int index = 0;index < half_odf;++index)
        {
            tipl::vector<3,float> displacement(handle->dir.odf_table[index]);
            if(odf_color == 0) // 0:directional
                displacement *= (std::fabs(odf_buffer[index])-odf_min)*odf_scale;
            else
            {
                if(odf_color == 1) // Blue: increased ODF
                    displacement *= std::max<float>(0.0f,odf_buffer[index])*odf_scale;
                else
                    // Red: decreased ODF
                    displacement *= -std::min<float>(0.0f,odf_buffer[index])*odf_scale;
            }
            iter[index] += displacement;
            iter[index+half_odf] -= displacement;
        }

        // calculate normal
        std::vector<tipl::vector<3,float> >::iterator iter2 = odf_norm.begin()+i*odf_dim;
        std::vector<tipl::vector<3,float> >::iterator end2 = iter2+odf_dim;
        for(int j = 0;j < handle->dir.odf_faces.size();++j)
        {
            unsigned short p1 = handle->dir.odf_faces[j][0];
            unsigned short p2 = handle->dir.odf_faces[j][1];
            unsigned short p3 = handle->dir.odf_faces[j][2];
            auto n = (iter[p1] - iter[p2]).cross_product(iter[p2] - iter[p3]);
            n.normalize();
            iter2[p1] += n;
            iter2[p2] += n;
            iter2[p3] += n;
        }
        for(;iter2 != end2;++iter2)
            iter2->normalize();
    });
}

void GLWidget::resizeGL(int width_, int height_)
{
    cur_width = width_ *  devicePixelRatio();
    cur_height = height_ *  devicePixelRatio();
}
void GLWidget::scale_by(float scalefactor)
{
    if(no_update)
        return;
    makeCurrent();
    glPushMatrix();
    glLoadIdentity();
    glScaled(scalefactor,scalefactor,scalefactor);
    if(edit_right && view_mode == view_mode_type::two)
    {
        glMultMatrixf(transformation_matrix2.begin());
        glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix2.begin());
    }
    else
    {
        glMultMatrixf(transformation_matrix.begin());
        glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix.begin());
        cur_tracking_window.ui->zoom_3d->setValue(std::pow(transformation_matrix.det(),1.0/3.0));
    }
    glPopMatrix();
}
void GLWidget::wheelEvent ( QWheelEvent * event )
{
    #ifdef QT6_PATCH
        auto x = event->position().x();
    #else
        auto x = event->x();
    #endif
    edit_right = (view_mode != view_mode_type::single && (x > cur_width / 2));
    double scalefactor = event->angleDelta().y();
    scalefactor /= 1200.0;
    scalefactor = 1.0+scalefactor;
    scale_by(scalefactor);
    update();
    event->ignore();

}

void GLWidget::slice_location(unsigned char dim,std::vector<tipl::vector<3,float> >& points)
{
    cur_tracking_window.current_slice->get_slice_positions(dim,points);
}

void GLWidget::get_view_dir(QPoint p,tipl::vector<3,float>& dir)
{
    tipl::matrix<4,4> m;
    float v[3];
    glGetFloatv(GL_PROJECTION_MATRIX,m.begin());
    // Compute the vector of the pick ray in screen space
    v[0] = (( 2.0f * (float(p.x()) * devicePixelRatio())/float(view_mode == view_mode_type::two ? cur_width/2:cur_width)) - 1 ) / m[0];
    v[1] = -(( 2.0f * (float(p.y()) * devicePixelRatio())/float(cur_height)) - 1 ) / m[5];
    v[2] = -1.0f;
    // Transform the screen space pick ray into 3D space
    dir[0] = v[0]*mat[0] + v[1]*mat[4] + v[2]*mat[8];
    dir[1] = v[0]*mat[1] + v[1]*mat[5] + v[2]*mat[9];
    dir[2] = v[0]*mat[2] + v[1]*mat[6] + v[2]*mat[10];
    dir.normalize();
}

float GLWidget::get_slice_projection_point(unsigned char dim,
                                const tipl::vector<3,float>& pos,
                                const tipl::vector<3,float>& dir,
                                float& dx,float& dy)
{
    std::vector<tipl::vector<3,float> > slice_points(4);
    slice_location(dim,slice_points);
    tipl::vector<3,float> pos_offset(pos),v1(slice_points[1]),v2(slice_points[2]),v3(dir);
    pos_offset -= slice_points[0];
    v1 -= slice_points[0];
    v2 -= slice_points[0];
    tipl::matrix<3,3,float> m;
    m[0] = v1[0];
    m[1] = v2[0];
    m[2] = -v3[0];
    m[3] = v1[1];
    m[4] = v2[1];
    m[5] = -v3[1];
    m[6] = v1[2];
    m[7] = v2[2];
    m[8] = -v3[2];

    if(!m.inv())
        return 0.0;
    pos_offset.rotate(m);
    dx = pos_offset[0];
    dy = pos_offset[1];
    return pos_offset[2];
}

tipl::vector<3,float> get_norm(const std::vector<tipl::vector<3,float> >& slice_points)
{
    tipl::vector<3,float> v1(slice_points[1]),v2(slice_points[2]),norm;
    v1 -= slice_points[0];
    v2 -= slice_points[0];
    norm = v1.cross_product(v2);
    norm.normalize();
    return norm;
}
bool GLWidget::select_object(void)
{
    device_selected = false;
    region_selected = false;
    slice_selected = false;

    object_distance = slice_distance = std::numeric_limits<float>::max();
    // select device to move
    if(get_param("show_device") && !cur_tracking_window.deviceWidget->devices.empty())
    {
        for(object_distance = 0.0f;object_distance < 2000.0f && !device_selected;object_distance += 1.0f)
        {
            tipl::vector<3,float> p(dir1);
            p *= object_distance;
            p += pos;
            float min_distance = std::numeric_limits<float>::max();
            float distance,slength;
            for(size_t index = 0;index < cur_tracking_window.deviceWidget->devices.size();++index)
                if(cur_tracking_window.deviceWidget->item(int(index),0)->checkState() == Qt::Checked &&
                   cur_tracking_window.deviceWidget->devices[index]->selected(p,
                            cur_tracking_window.handle->vs[0],slength,distance) &&
                   distance < min_distance)
                {
                    min_distance = distance;
                    device_selected_length = slength;
                    selected_index = index;
                    device_selected = true;
                    slice_distance = object_distance;
                }
        }
    }
    // select region to move
    if(get_param("show_region") && !cur_tracking_window.regionWidget->regions.empty() && !device_selected)
    {
        // select object
        for(object_distance = 0.0f;object_distance < 2000.0f && !region_selected;object_distance += 1.0f)
        {
            tipl::vector<3,float> p(dir1);
            p *= object_distance;
            p += pos;
            tipl::vector<3,short> voxel(p);
            if(!cur_tracking_window.handle->dim.is_valid(voxel))
                continue;
            tipl::par_for(cur_tracking_window.regionWidget->regions.size(),[&](size_t index)
            {
                if(region_selected ||
                   cur_tracking_window.regionWidget->item(int(index),0)->checkState() != Qt::Checked)
                    return;
                if(cur_tracking_window.regionWidget->regions[index]->has_point(voxel) && !region_selected)
                {
                    region_selected = true;
                    selected_index = index;
                    slice_distance = object_distance;
                }
            });
        }
    }
    // select slices to move
    if(get_param("show_slice"))
    {
        bool show_slice[3];
        show_slice[0] = cur_tracking_window.ui->glSagCheck->checkState();
        show_slice[1] = cur_tracking_window.ui->glCorCheck->checkState();
        show_slice[2] = cur_tracking_window.ui->glAxiCheck->checkState();
        // now check whether the slices are selected
        for(unsigned char dim = 0;dim < 3;++dim)
        {
            if(!show_slice[dim])
                continue;
            float d = get_slice_projection_point(dim,pos,dir1,slice_dx,slice_dy);
            if(slice_dx > 0.0f && slice_dy > 0.0f &&
               slice_dx < 1.0f && slice_dy < 1.0f &&
                    d > 0 && d < slice_distance)
            {
                moving_at_slice_index = dim;
                slice_distance = d;
                slice_selected = true;
            }
        }
    }
    return region_selected || slice_selected || device_selected;
}


void GLWidget::get_pos(void)
{
    //glMultMatrixf(transformation_matrix);
    glGetFloatv(GL_MODELVIEW_MATRIX,mat.begin());
    tipl::matrix<4,4> view = (edit_right) ? transformation_matrix2*mat : transformation_matrix*mat;
    mat = tipl::inverse(view);
    pos[0] = mat[12];
    pos[1] = mat[13];
    pos[2] = mat[14];
}

void GLWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    makeCurrent();
    QPoint cur_pos = convert_pos(event);
    get_pos();
    get_view_dir(cur_pos,dir1);
    if(!select_object())
        return;
    if(event->button() == Qt::LeftButton)
    {
        if(region_selected)
        {
            cur_tracking_window.regionWidget->setCurrentCell(selected_index,0);
            cur_tracking_window.regionWidget->move_slice_to_current_region();
            emit region_edited();
        }
        if(device_selected)
        {
            cur_tracking_window.deviceWidget->setCurrentCell(selected_index,0);
        }
        if(slice_selected)
        {
            switch(moving_at_slice_index)
            {
            case 0:
                cur_tracking_window.ui->glSagCheck->setChecked(false);
                break;
            case 1:
                cur_tracking_window.ui->glCorCheck->setChecked(false);
                break;
            case 2:
                cur_tracking_window.ui->glAxiCheck->setChecked(false);
                break;
            }
        }
    }
}

bool GLWidget::get_mouse_pos(QMouseEvent *event,tipl::vector<3,float>& position)
{
    makeCurrent();
    QPoint cur_pos = event->pos();
    if(edit_right)
        cur_pos.setX(cur_pos.x() - cur_width / 2);

    get_pos();
    tipl::vector<3,float> cur_dir;
    get_view_dir(cur_pos,cur_dir);

    bool show_slice[3];
    show_slice[0] = cur_tracking_window.ui->glSagCheck->checkState();
    show_slice[1] = cur_tracking_window.ui->glCorCheck->checkState();
    show_slice[2] = cur_tracking_window.ui->glAxiCheck->checkState();
    // select slice
    {
        // now check whether the slices are selected
        std::vector<float> x(3),y(3),d(3);
        for(unsigned char dim = 0;dim < 3;++dim)
        {
            if(!show_slice[dim])
                continue;
            d[dim] = get_slice_projection_point(dim,pos,cur_dir,x[dim],y[dim]);
            if(d[dim] == 0.0 || x[dim] < 0.0 || x[dim] > 1.0 || y[dim] < 0.0 || y[dim] > 1.0)
                d[dim] = std::numeric_limits<float>::max();
        }
        unsigned int min_index = std::min_element(d.begin(),d.end())-d.begin();
        if(d[min_index] != 0.0 && d[min_index] != std::numeric_limits<float>::max())
        {
            std::vector<tipl::vector<3,float> > points(4);
            slice_location(min_index,points);
            position = points[0] + (points[1]-points[0])*x[min_index] + (points[2]-points[0])*y[min_index];
            return true;
        }
    }
    return false;
}
QPoint GLWidget::convert_pos(QMouseEvent *event)
{
    QPoint p = event->pos();
    if(edit_right)
        p.setX(p.x() - cur_width / 2);
    if(view_mode != view_mode_type::single)
        p.setX(p.x()*2);
    return p;
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    makeCurrent();
    setFocus();// for key stroke to work
    edit_right = (view_mode != view_mode_type::single && (event->pos().x() > cur_width / 2));
    lastPos = curPos = convert_pos(event);
    if(editing_option != none)
        get_pos();
    if(editing_option == selecting)
        {
            dirs.clear();
            last_select_point = lastPos;
            dirs.push_back(tipl::vector<3,float>());
            get_view_dir(last_select_point,dirs.back());
        }
    else
        if(editing_option == moving)
        {
            get_view_dir(lastPos,dir1);
            dir1.normalize();
            if(!select_object())
            {
                editing_option = none;
                setCursor(Qt::ArrowCursor);
                return;
            }
            // if only slice is selected or slice is at the front, then move slice
            // if the slice is the picture, then the slice will be moved.
            if(slice_selected && object_distance > slice_distance)
            {
                if(!dynamic_cast<CustomSliceModel*>(cur_tracking_window.current_slice.get()) ||
                   dynamic_cast<CustomSliceModel*>(cur_tracking_window.current_slice.get())->picture.empty())
                {
                    editing_option = dragging;
                    return;
                }
            }
            if(region_selected)
                cur_tracking_window.regionWidget->selectRow(int(selected_index));
            if(device_selected)
                cur_tracking_window.deviceWidget->selectRow(int(selected_index));
            // determine the moving direction of the region
            float angle[3] = {0,0,0};
            for(unsigned char dim = 0;dim < 3;++dim)
            {
                std::vector<tipl::vector<3,float> > points(4);
                slice_location(dim,points);
                angle[dim] = std::fabs(dir1*get_norm(points));
            }
            moving_at_slice_index = std::max_element(angle,angle+3)-angle;
            if(get_slice_projection_point(moving_at_slice_index,pos,dir1,slice_dx,slice_dy) == 0.0)
            {
                editing_option = none;
                setCursor(Qt::ArrowCursor);
                return;
            }
            accumulated_dis = tipl::zero<float>();
            update();
        }
}
void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    makeCurrent();
    if(editing_option == selecting)
    {
        last_select_point = convert_pos(event);
        dirs.push_back(tipl::vector<3,float>());
        get_view_dir(last_select_point,dirs.back());
        angular_selection = event->button() == Qt::RightButton;
        emit edited();
    }
    if(device_selected)
    {
        device_selected = false;
        update();
    }
    editing_option = none;
    setCursor(Qt::ArrowCursor);
}
void GLWidget::move_by(int x,int y)
{
    makeCurrent();
    glPushMatrix();
    glLoadIdentity();
    glTranslated(x/5.0,y/5.0,0);
    if(edit_right && view_mode == view_mode_type::two)
    {
        glMultMatrixf(transformation_matrix2.begin());
        glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix2.begin());
    }
    else
    {
        glMultMatrixf(transformation_matrix.begin());
        glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix.begin());
    }
    glPopMatrix();
    update();
}
void handle_rotate(bool circular,bool only_y,float fx,float fy,float dx,float dy)
{
    if(circular)
    {
        if(fx < 0)
            dy = -dy;
        if(fy > 0)
            dx = -dx;
        glRotatef((dy+dx)/4.0f, 0.0f, 0.0f, 1.0f);
    }
    else
    {
        glRotatef(dy / 2.0f, 1.0f, 0.0f, 0.0f);
        if(!only_y)
            glRotatef(-dx / 2.0f, 0.0f, 1.0f, 0.0f);
    }
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    makeCurrent();
    curPos = convert_pos(event);
    if(editing_option == selecting)
    {
        if(event->modifiers() & Qt::ShiftModifier)
        {
            QPoint dis(curPos);
            dis -= last_select_point;
            if(dis.manhattanLength() < 20)
                return;
            last_select_point = curPos;
            dirs.push_back(tipl::vector<3,float>());
            get_view_dir(last_select_point,dirs.back());
        }
        update();
        return;
    }

    if(editing_option == moving)
    {
        std::vector<tipl::vector<3,float> > points(4);
        slice_location(moving_at_slice_index,points);
        get_view_dir(curPos,dir2);
        float dx,dy;
        if(get_slice_projection_point(moving_at_slice_index,pos,dir2,dx,dy) == 0.0f)
            return;
        tipl::vector<3,float> v1(points[1]),v2(points[2]),dis;
        v1 -= points[0];
        v2 -= points[0];
        dis = v1*(dx-slice_dx)+v2*(dy-slice_dy);
        dis -= accumulated_dis;
        if(device_selected && selected_index < cur_tracking_window.deviceWidget->devices.size())
        {
            cur_tracking_window.deviceWidget->devices[selected_index]->move(device_selected_length,dis);
            accumulated_dis += dis;
            update();
            return;
        }

        if(region_selected && selected_index < cur_tracking_window.regionWidget->regions.size())
        {
            if(cur_tracking_window.regionWidget->regions[selected_index]->shift(dis))
            {
                emit region_edited();
                accumulated_dis += dis;
            }
            return;
        }

        // a picture slice is selected
        if(slice_selected && dynamic_cast<CustomSliceModel*>(cur_tracking_window.current_slice.get()))
        {
            auto slice = dynamic_cast<CustomSliceModel*>(cur_tracking_window.current_slice.get());
            if (event->buttons() & Qt::LeftButton)
            {
                slice->arg_min.translocation[0] += dis[0]*0.05f;
                slice->arg_min.translocation[1] += dis[1]*0.05f;
                slice->arg_min.translocation[2] += dis[2]*0.05f;
            }
            else
            {
                slice->arg_min.scaling[0] += dis[0]*0.005f;
                slice->arg_min.scaling[1] += dis[1]*0.005f;
                slice->arg_min.scaling[2] += dis[2]*0.005f;
            }
            emit region_edited();
            slice->update_transform();
            update();
            accumulated_dis += dis;
        }
        return;
    }
    // move slice
    if(editing_option == dragging)
    {
        std::vector<tipl::vector<3,float> > points(4);
        slice_location(moving_at_slice_index,points);
        get_view_dir(curPos,dir2);
        float move_dis = (dir2-dir1)*get_norm(points);
        move_dis *= slice_distance;
        if(std::fabs(move_dis) < 1.0f)
            return;
        move_dis = std::round(move_dis);
        switch(moving_at_slice_index)
        {
        case 0:
            cur_tracking_window.ui->glSagBox->setValue(cur_tracking_window.ui->glSagBox->value()+move_dis);
            break;
        case 1:
            cur_tracking_window.ui->glCorBox->setValue(cur_tracking_window.ui->glCorBox->value()-move_dis);
            break;
        case 2:
            cur_tracking_window.ui->glAxiBox->setValue(cur_tracking_window.ui->glAxiBox->value()+move_dis);
            break;
        }
        dir1 = dir2;
        return;
    }


    float dx = curPos.x() - lastPos.x();
    float dy = curPos.y() - lastPos.y();
    if(event->modifiers() & Qt::ControlModifier)
    {
        dx /= 16.0f;
        dy /= 16.0f;
    }

    glPushMatrix();
    glLoadIdentity();
    // left button down
    if ((event->buttons() & Qt::LeftButton) && !(event->buttons() & Qt::RightButton) && !(event->modifiers() && Qt::ShiftModifier))
    {
        auto& tran = (edit_right && view_mode == view_mode_type::two) ? transformation_matrix2:transformation_matrix;
        auto& rot = (edit_right && view_mode == view_mode_type::two) ? rotation_matrix2:rotation_matrix;

        float fx = (float)lastPos.x()/(float)width()-0.5f;
        float fy = (float)lastPos.y()/(float)height()-0.5f;
        bool circular_rotate = fx*fx+fy*fy > 0.25;
        bool only_y = (event->modifiers() & Qt::ShiftModifier);
        handle_rotate(circular_rotate,only_y,fx,fy,dx,dy);
        glMultMatrixf(tran.begin());
        glGetFloatv(GL_MODELVIEW_MATRIX,tran.begin());
        // book keeping the rotation matrix
        glLoadIdentity();
        handle_rotate(circular_rotate,only_y,fx,fy,dx,dy);
        glMultMatrixf(rot.begin());
        glGetFloatv(GL_MODELVIEW_MATRIX,rot.begin());
    }
    else
    {
        // right button
        if ((event->buttons() & Qt::RightButton) && !(event->buttons() & Qt::LeftButton) && !(event->modifiers() && Qt::ShiftModifier))
        {
            double scalefactor = (-dx-dy+100.0)/100.0;
            glScaled(scalefactor,scalefactor,scalefactor);
        }
        else
        // middle button down or right/left both down
            glTranslated(dx/5.0,dy/5.0,0);

        if(edit_right && view_mode == view_mode_type::two)
        {
            glMultMatrixf(transformation_matrix2.begin());
            glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix2.begin());
        }
        else
        {
            glMultMatrixf(transformation_matrix.begin());
            glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix.begin());
            if(event->buttons() & Qt::RightButton)
                cur_tracking_window.ui->zoom_3d->setValue(std::pow(transformation_matrix.det(),1.0/3.0));
        }
    }


    glPopMatrix();
    update();
    lastPos = curPos;
}

void GLWidget::saveCamera(void)
{
    QString filename = cur_tracking_window.
            get_save_file_name("Save Translocation Matrix","camera.txt","Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    command("save_camera",filename);
}

void GLWidget::loadCamera(void)
{
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Open Translocation Matrix",QFileInfo(cur_tracking_window.work_path).absolutePath(),"Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    command("load_camera",filename);
}
void GLWidget::addSurface(void)
{
    command("add_surface",qobject_cast<QAction*>(sender())->text());
}

void GLWidget::copyToClipboard(void)
{
    QApplication::clipboard()->setImage(tipl::qt::get_bounding_box(grab_image()));
}

void GLWidget::copyToClipboardEach(QTableWidget* widget,unsigned int col_size)
{
    std::vector<bool> is_checked(uint32_t(widget->rowCount()));
    for (int i = 0;i < widget->rowCount();++i)
    {
        is_checked[size_t(i)] = (widget->item(i,0)->checkState() == Qt::Checked);
        widget->item(i,0)->setCheckState(Qt::Unchecked);
    }
    std::vector<QImage> images;
    for (int i = 0;i < widget->rowCount();++i)
        if(is_checked[uint32_t(i)])
        {
            widget->item(i,0)->setCheckState(Qt::Checked);
            images.push_back(tipl::qt::get_bounding_box(grab_image()));
            if(images.back().width() == 0)
                images.pop_back();
            widget->item(i,0)->setCheckState(Qt::Unchecked);
        }
    if(images.empty())
    {
        QMessageBox::critical(this,"ERROR","No visible output captured. Did you check any region or tract?");
        return;
    }
    QApplication::clipboard()->setImage(tipl::qt::create_mosaic(images,col_size));
    for (int i = 0;i < widget->rowCount();++i)
        if(is_checked[uint32_t(i)])
            widget->item(i,0)->setCheckState(Qt::Checked);
    QMessageBox::information(this,"DSI Studio","Images captured to clipboard");

}

void GLWidget::copyToClipboardEachTract(void)
{
    bool ok = true;
    int col_count = QInputDialog::getInt(this,"DSI Studio","Column Count",5,1,50,1&ok);
    if(!ok)
        return;
    copyToClipboardEach(cur_tracking_window.tractWidget,uint32_t(col_count));
}

void GLWidget::copyToClipboardEachRegion(void)
{
    bool ok = true;
    int col_count = QInputDialog::getInt(this,"DSI Studio","Column Count",5,1,50,1&ok);
    if(!ok)
        return;
    copyToClipboardEach(cur_tracking_window.regionWidget,uint32_t(col_count));
}


void GLWidget::get3View(QImage& I,unsigned int type)
{
    makeCurrent();
    set_view_flip = false;
    set_view(0);
    QImage image0 = grab_image();
    set_view_flip = true;
    set_view(0);
    QImage image00 = grab_image();
    set_view_flip = true;
    set_view(1);
    QImage image1 = grab_image();
    set_view_flip = true;
    set_view(2);
    QImage image2 = grab_image();

    if(type == 0)
    {
        QImage image3 = cur_tracking_window.scene.view_image.scaledToWidth(image0.width()).convertToFormat(QImage::Format_RGB32);
        QImage all(image0.width()*2,image0.height()*2,QImage::Format_RGB32);
        QPainter painter(&all);
        painter.drawImage(0,0,image0);
        painter.drawImage(image0.width(),0,image1);
        painter.drawImage(image0.width(),image0.height(),image2);
        painter.drawImage(0,image0.height(),image3);
        I = all;
    }
    if(type == 1) // horizontal
    {
        image0 = tipl::qt::get_bounding_box(image0);
        image00 = tipl::qt::get_bounding_box(image00);
        image1 = tipl::qt::get_bounding_box(image1);
        image2 = tipl::qt::get_bounding_box(image2);
        int height_shift = (image2.height()-image0.height())/2;
        QImage all(image0.width()+image00.width()+image1.width()+image2.width(),image2.height(),QImage::Format_RGB32);
        all.fill(image0.pixel(0,0));
        QPainter painter(&all);
        painter.setCompositionMode(QPainter::CompositionMode_Source);
        painter.drawImage(0,height_shift,image0);
        painter.drawImage(image0.width(),height_shift,image00);
        painter.drawImage(image0.width()+image00.width(),height_shift,image1);
        painter.drawImage(image0.width()+image00.width()+image1.width(),0,image2);
        I = all;
    }
    if(type == 2)
    {
        QImage all(image0.width(),image0.height()*4,QImage::Format_RGB32);
        QPainter painter(&all);
        painter.drawImage(0,0,image0);
        painter.drawImage(0,image0.height(),image00);
        painter.drawImage(0,image0.height()*2,image1);
        painter.drawImage(0,image0.height()*3,image2);
        I = all;
    }
}
extern bool has_gui;
bool GLWidget::command(QString cmd,QString param,QString param2)
{
    if(cmd == "save_camera")
    {
        std::ofstream(param.toStdString().c_str()) << transformation_matrix;
    }
    if(cmd == "load_camera")
    {
        std::ifstream in(param.toStdString().c_str());
        if(in)
        {
            std::vector<float> data;
            std::copy(std::istream_iterator<float>(in),
                      std::istream_iterator<float>(),std::back_inserter(data));
            data.resize(16);
            std::copy(data.begin(),data.end(),transformation_matrix.begin());
            update();
        }
        return true;
    }
    if(cmd == "set_zoom")
    {
        if(param.isEmpty())
            return true;
        double zoom = param.toFloat();
        if(zoom == 0)
            return true;
        zoom /= std::pow(transformation_matrix.det(),1.0/3.0);

        if(zoom < 0.99 || zoom > 1.01)
        {
            scale_by(zoom);
            update();
        }
        return true;
    }
    if(cmd == "set_view")
    {
        if(param.isEmpty())
            return true;
        makeCurrent();
        set_view(param.toInt());
        update();
        return true;
    }
    if(cmd == "set_stereoscopic")
    {
        makeCurrent();
        view_mode = GLWidget::view_mode_type::stereo;
        update();
        return true;
    }
    if(cmd == "move_slice")
    {
        switch(param.toInt())
        {
        case 0:
            cur_tracking_window.ui->glSagSlider->setValue(param2.toInt());
            break;
        case 1:
            cur_tracking_window.ui->glCorSlider->setValue(param2.toInt());
            break;
        case 2:
            cur_tracking_window.ui->glAxiSlider->setValue(param2.toInt());
            break;
        }
        return true;
    }
    if(cmd == "slice_off")
    {
        if(param.isEmpty())
        {
            cur_tracking_window.ui->glCorCheck->setChecked(false);
            cur_tracking_window.ui->glSagCheck->setChecked(false);
            cur_tracking_window.ui->glAxiCheck->setChecked(false);
            return true;
        }
        switch(param.toInt())
        {
        case 0:
            cur_tracking_window.ui->glSagCheck->setChecked(false);
            break;
        case 1:
            cur_tracking_window.ui->glCorCheck->setChecked(false);
            break;
        case 2:
            cur_tracking_window.ui->glAxiCheck->setChecked(false);
            break;
        }
        update();
        return true;

    }
    if(cmd == "slice_on")
    {
        if(param.isEmpty())
        {
            cur_tracking_window.ui->glCorCheck->setChecked(true);
            cur_tracking_window.ui->glSagCheck->setChecked(true);
            cur_tracking_window.ui->glAxiCheck->setChecked(true);
            return true;
        }
        switch(param.toInt())
        {
        case 0:
            cur_tracking_window.ui->glSagCheck->setChecked(true);
            break;
        case 1:
            cur_tracking_window.ui->glCorCheck->setChecked(true);
            break;
        case 2:
            cur_tracking_window.ui->glAxiCheck->setChecked(true);
            break;
        }
        update();
        return true;
    }
    if(cmd == "add_surface")
    {
        tipl::image<3> crop_image;
        float resolution_ratio = 1.0;
        bool is_wm = (cur_tracking_window.current_slice->get_name().find("wm") != std::string::npos);
        float threshold = 25.0f;
        CustomSliceModel* reg_slice = dynamic_cast<CustomSliceModel*>(cur_tracking_window.current_slice.get());
        if(reg_slice)
        {
            if(!reg_slice->skull_removed_images.empty())
                crop_image = reg_slice->skull_removed_images;
        }
        else
        {
            // use ICBM152 wm as the surface
            tipl::io::gz_nifti nifti;
            if(nifti.load_from_file(cur_tracking_window.handle->wm_template_file_name.c_str()))
            {
                tipl::matrix<4,4,float> trans;
                nifti.toLPS(crop_image);
                nifti.get_image_transformation(trans);
                if(cur_tracking_window.handle->mni2sub(crop_image,trans))
                    is_wm = true;
                else
                    crop_image.clear();
            }
        }
        if(crop_image.empty())
            crop_image = cur_tracking_window.current_slice->get_source();
        if(!is_wm)
            threshold = tipl::segmentation::otsu_threshold(crop_image)*1.25f;

        if(!param2.isEmpty())
            threshold = param2.toFloat();
        else
        if(has_gui)
        {
            bool ok;
            threshold = float(QInputDialog::getDouble(this,
                "DSI Studio","Threshold:", double(threshold),
                double(tipl::min_value(crop_image)),
                double(tipl::max_value(crop_image)),
                4, &ok));
            if (!ok)
                return true;
        }

        {
            surface = std::make_shared<RegionRender>();
            if(!param.isEmpty() && !param.toLower().contains("full"))
            {
                tipl::image<3,unsigned char> remain_part(crop_image.shape());
                if(param.toLower().contains("left"))
                {
                    for(unsigned int index = 0;index < remain_part.size();index += remain_part.width())
                    {
                        std::fill(remain_part.begin()+index+cur_tracking_window.current_slice->slice_pos[0],
                                  remain_part.begin()+index+remain_part.width(),1);
                    }
                }
                if(param.toLower().contains("right"))
                {
                    for(unsigned int index = 0;index < remain_part.size();index += remain_part.width())
                    {
                        std::fill(remain_part.begin()+index,
                                  remain_part.begin()+index+cur_tracking_window.current_slice->slice_pos[0],1);
                    }
                }
                if(param.toLower().contains("upper"))
                {
                    std::fill(remain_part.begin()+cur_tracking_window.current_slice->slice_pos[2]*remain_part.plane_size(),
                              remain_part.end(),1);
                }
                if(param.toLower().contains("lower"))
                {
                    std::fill(remain_part.begin(),
                              remain_part.begin()+cur_tracking_window.current_slice->slice_pos[2]*remain_part.plane_size(),1);
                }
                if(param.toLower().contains("posterior"))
                {
                    for(unsigned int index = 0;index < remain_part.size();index += remain_part.plane_size())
                    {
                        std::fill(remain_part.begin()+index+int64_t(cur_tracking_window.current_slice->slice_pos[1])*remain_part.width(),
                                  remain_part.begin()+index+int64_t(remain_part.plane_size()),1);
                    }
                }
                if(param.toLower().contains("anterior"))
                {
                    for(unsigned int index = 0;index < remain_part.size();index += remain_part.plane_size())
                    {
                        std::fill(remain_part.begin()+index,
                                  remain_part.begin()+index+int64_t(cur_tracking_window.current_slice->slice_pos[1])*remain_part.width(),1);
                    }
                }
                crop_image *= remain_part;
            }


            switch(get_param("surface_mesh_smoothed"))
            {
            case 1:
                tipl::filter::gaussian(crop_image);
                break;
            case 2:
                {
                tipl::image<3,unsigned char> mask(crop_image);
                for(size_t index = 0;index < crop_image.size();++index)
                    if(crop_image[index] > threshold)
                        mask[index] = 1;
                tipl::morphology::defragment(mask);
                tipl::morphology::negate(mask);
                tipl::morphology::defragment(mask);
                tipl::morphology::negate(mask);
                tipl::morphology::smoothing_mt(mask);
                tipl::morphology::dilation_mt(mask);
                for(size_t index = 0;index < crop_image.size();++index)
                    if(mask[index] == 0)
                        crop_image[index] *= 0.2f;
                tipl::filter::gaussian(crop_image);
                }
                break;
            }
            if(!surface->load(crop_image,threshold))
            {
                surface.reset();
                return true;
            }
        }

        if(!cur_tracking_window.current_slice->is_diffusion_space)
            surface->transform_point_list(cur_tracking_window.current_slice->to_dif);

        update();
        return true;
    }
    if(cmd == "save_image")
    {
        if(param.isEmpty())
            param = QFileInfo(cur_tracking_window.windowTitle()).fileName()+".image.jpg";
        if(!param2.isEmpty())
        {
            std::istringstream in(param2.toStdString().c_str());
            int w = 0;
            int h = 0;
            int ow = width(),oh = height();
            in >> w >> h;
            resize(w,h);
            resizeGL(w,h);
            grab_image().save(param);
            resize(ow,oh);
            resizeGL(ow,oh);
        }
        else
            grab_image().save(param);
        return true;
    }
    if(cmd == "save_3view_image")
    {
        if(param.isEmpty())
            param = QFileInfo(cur_tracking_window.windowTitle()).completeBaseName()+".3view_image.jpg";
        QImage all;
        get3View(all,0);
        all.save(param);
        return true;
    }
    if(cmd == "save_h3view_image")
    {
        if(param.isEmpty())
            param = QFileInfo(cur_tracking_window.windowTitle()).completeBaseName()+".h3view_image.jpg";
        QImage all;
        get3View(all,1);
        all.save(param);
        return true;
    }
    if(cmd == "save_v3view_image")
    {
        if(param.isEmpty())
            param = QFileInfo(cur_tracking_window.windowTitle()).completeBaseName()+".v3view_image.jpg";
        QImage all;
        get3View(all,2);
        all.save(param);
        return true;
    }
    if(cmd == "save_rotation_video")
    {
        if(param.isEmpty())
            param = QFileInfo(cur_tracking_window.windowTitle()).completeBaseName()+".rotation_movie.avi";
        if(QFileInfo(param).suffix() == "avi")
        {
            tipl::progress prog("save video");
            int ow = width(),oh = height();
            tipl::io::avi avi;
            #ifndef __APPLE__
                resize(1980,1080);
                resizeGL(1980,1080);
            #endif
            for(float index = 0.0f;prog(index,360);index += 0.2f)
            {
                rotate_angle(0.2f,0,1.0,0.0);
                QBuffer buffer;
                QImageWriter writer(&buffer, "JPG");
                QImage I = grab_image();
                writer.write(I);
                if(index == 0.0f)
                    avi.open(param.toStdString().c_str(),I.width(),I.height(), "MJPG", 30/*fps*/);
                QByteArray data = buffer.data();
                avi.add_frame(data.begin(),uint32_t(data.size()),true);
            }
            avi.close();
            resize(ow,oh);
        }
        else
        {
            tipl::progress prog_("save image");
            float angle = (param2.isEmpty()) ? 1 : param2.toFloat();
            for(float index = 0;prog_(index,360);index += angle)
            {
                QString file_name = QFileInfo(param).absolutePath()+"//"+
                        QFileInfo(param).completeBaseName()+"_"+QString::number(index)+"."+
                        QFileInfo(param).suffix();
                tipl::out() << file_name.toStdString() << std::endl;
                rotate_angle(angle,0,1.0,0.0);
                QImage I = grab_image();
                I.save(file_name);
            }
        }
        return true;
    }
    return false;
}
void GLWidget::catchScreen(void)
{
    QString filename = QFileDialog::getSaveFileName(
               this,
               "Save Images files",
               QFileInfo(cur_tracking_window.windowTitle()).completeBaseName()+".image.jpg",
               "Image files (*.png *.bmp *.jpg *.tif);;All files (*)");
    if(filename.isEmpty())
        return;
    command("save_image",filename);
}

void GLWidget::catchScreen2(void)
{
    bool ok;
    QString result = QInputDialog::getText(this,"DSI Studio","Assign image dimension (width height)",QLineEdit::Normal,
                                           QString::number(cur_width)+" "+QString::number(cur_height),&ok);
    if(!ok)
        return;
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save Images files",
            QFileInfo(cur_tracking_window.windowTitle()).completeBaseName()+".hd_image.jpg",
            "Image files (*.png *.bmp *.jpg *.tif);;All files (*)");
    if(filename.isEmpty())
        return;
    command("save_image",filename,result);
}

void GLWidget::save3ViewImage(void)
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Assign image name",
            QFileInfo(cur_tracking_window.windowTitle()).completeBaseName()+".3view_image.jpg",
            "Image files (*.png *.bmp *.jpg *.tif);;All files (*)");
    command("save_3view_image",filename);

}


void GLWidget::saveRotationSeries(void)
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                "Assign video name",
                QFileInfo(cur_tracking_window.windowTitle()).completeBaseName()+".rotation_movie.avi",
                "Video file (*.avi);;Image filess (*.jpg *.png);;All files (*)");
    if(filename.isEmpty())
        return;
    command("save_rotation_video",filename);
}

void GLWidget::rotate_angle(float angle,float x,float y,float z)
{
    makeCurrent();
    glPushMatrix();

    glLoadIdentity();
    glRotated(angle,x,y,z);
    glMultMatrixf(rotation_matrix.begin());
    glGetFloatv(GL_MODELVIEW_MATRIX,rotation_matrix.begin());

    glLoadIdentity();
    glRotated(angle,x,y,z);
    glMultMatrixf(transformation_matrix.begin());
    glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix.begin());

    glLoadIdentity();
    glRotated(angle,x,y,z);
    glMultMatrixf(rotation_matrix2.begin());
    glGetFloatv(GL_MODELVIEW_MATRIX,rotation_matrix2.begin());

    glLoadIdentity();
    glRotated(angle,x,y,z);
    glMultMatrixf(transformation_matrix2.begin());
    glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix2.begin());
    glPopMatrix();

    update();
}

void GLWidget::rotate(void)
{
    int now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-time).count();
    rotate_angle((now_time-last_time)/100.0,0,1.0,0.0);
    last_time = now_time;
    update();
}
void GLWidget::record_video(void)
{
    if(video_handle.get())
    {
        video_timer->stop();
        video_handle->close();
        QMessageBox::information(this,"DSI Studio","Video saved");
        video_handle.reset();
    }
    else
    {
        QString file = QFileDialog::getSaveFileName(
                                    this,
                                    "Save video",
                                    QFileInfo(cur_tracking_window.windowTitle()).completeBaseName()+".avi",
                                    "Report file (*.avi);;All files (*)");
        if(file.isEmpty())
            return;
        QMessageBox::information(this,"DSI Studio","Press Ctrl+Shift+R again to stop recording.");
        QImage I = grabFramebuffer();
        video_handle = std::make_shared<tipl::io::avi>();
        video_handle->open(file.toStdString().c_str(),I.width(),I.height(), "MJPG", 10/*fps*/);
        video_frames = 0;
        video_timer = std::make_shared<QTimer>(this);
        video_timer->stop();
        connect(video_timer.get(),SIGNAL(timeout()),this,SLOT(update()));
        video_timer->setInterval(100);
        video_timer->start(100);
    }
}



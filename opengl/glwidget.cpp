#define NOMINMAX
#include <QtOpenGL>
#include <QtGui>
#include <QMessageBox>
#include <QInputDialog>
#include <QSettings>
#include <QTimer>
#include <QClipboard>
#include <vector>
#include "glwidget.h"
#include "tracking/tracking_window.h"
#include "ui_tracking_window.h"
#include "renderingtablewidget.h"
#include "tracking/region/regiontablewidget.h"
#include "SliceModel.h"
#include "fib_data.hpp"
#include "tracking/color_bar_dialog.hpp"
#include "libs/tracking/tract_model.hpp"

GLenum BlendFunc1[] = {GL_ZERO,GL_ONE,GL_DST_COLOR,
                      GL_ONE_MINUS_DST_COLOR,GL_SRC_ALPHA,
                      GL_ONE_MINUS_SRC_ALPHA,GL_DST_ALPHA,
                      GL_ONE_MINUS_DST_ALPHA};
GLenum BlendFunc2[] = {GL_ZERO,GL_ONE,GL_SRC_COLOR,
                      GL_ONE_MINUS_DST_COLOR,GL_SRC_ALPHA,
                      GL_ONE_MINUS_SRC_ALPHA,GL_DST_ALPHA,
                      GL_ONE_MINUS_DST_ALPHA};

GLWidget::GLWidget(bool samplebuffer,
                   tracking_window& cur_tracking_window_,
                   RenderingTableWidget* renderWidget_,
                   QWidget *parent)
                       : QGLWidget(samplebuffer ? QGLFormat(QGL::SampleBuffers):QGLFormat(),parent),
        cur_tracking_window(cur_tracking_window_),
        renderWidget(renderWidget_),
        tracts(0),
        cur_height(1),
        cur_width(1),
        editing_option(none),
        set_view_flip(false),
        view_mode(view_mode_type::single)
{
    std::fill(slice_texture,slice_texture+3,0);
    odf_dim = 0;
    odf_slide_pos = 0;
    slice_pos[0] = slice_pos[1] = slice_pos[2] = -1;
    transformation_matrix.identity();
    rotation_matrix.identity();
    transformation_matrix2.identity();
    rotation_matrix2.identity();
}

GLWidget::~GLWidget()
{
    makeCurrent();
    deleteTexture(slice_texture[0]);
    deleteTexture(slice_texture[1]);
    deleteTexture(slice_texture[2]);
    glDeleteLists(tracts, 1);
    //std::cout << __FUNCTION__ << " " << __FILE__ << std::endl;
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
    int v = renderWidget->getData(name).toInt();
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



void check_error(const char* line)
{
    GLenum code;
    while(code = glGetError())
    {
        std::cout << line << std::endl;
        switch(code)
        {
        case GL_INVALID_ENUM:
            std::cout << "GL_INVALID_ENUM" << std::endl;
            break;
        case GL_INVALID_VALUE:
            std::cout << "GL_INVALID_VALUE" << std::endl;
            break;
        case GL_INVALID_OPERATION:
            std::cout << "GL_INVALID_OPERATION" << std::endl;
            break;
        case GL_STACK_OVERFLOW:
            std::cout << "GL_STACK_OVERFLOW" << std::endl;
            break;
        case GL_STACK_UNDERFLOW:
            std::cout << "GL_STACK_UNDERFLOW" << std::endl;
            break;
        case GL_OUT_OF_MEMORY:
            std::cout << "GL_OUT_OF_MEMORY" << std::endl;
            break;

        }
    }
}


void GLWidget::set_view(unsigned char view_option)
{
    float scale = std::pow(transformation_matrix.det(),1.0/3.0);
    // initialize world matrix
    transformation_matrix.identity();
    rotation_matrix.identity();


    if(get_param("scale_voxel") && cur_tracking_window.handle->vs[0] > 0.0)
    {
        transformation_matrix[5] = cur_tracking_window.handle->vs[1] / cur_tracking_window.handle->vs[0];
        transformation_matrix[10] = cur_tracking_window.handle->vs[2] / cur_tracking_window.handle->vs[0];
    }

    image::vector<3,float> center_point(cur_tracking_window.handle->dim[0]/2.0-0.5,
                                        cur_tracking_window.handle->dim[1]/2.0-0.5,
                                        cur_tracking_window.handle->dim[2]/2.0-0.5);
    transformation_matrix[0] *= scale;
    transformation_matrix[5] *= scale;
    transformation_matrix[10] *= scale;
    transformation_matrix[12] = -transformation_matrix[0]*center_point[0];
    transformation_matrix[13] = -transformation_matrix[5]*center_point[1];
    transformation_matrix[14] = -transformation_matrix[10]*center_point[2];
    image::matrix<4,4,float> m;
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
    float angle_shift = 3.1415926*2.0/(light_option+1.0);

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
    GLfloat material2[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
    std::fill(material2,material2+3,emission);
    glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,material2);

    std::fill(material2,material2+3,specular);
    glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,material2);

    glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
    check_error(__FUNCTION__);
}

unsigned char getCurView(const image::matrix<4,4,float>& m)
{
    unsigned char cur_view = 0;
    {
        const float view_dirs[6][3] = {{1,0,0},{0,1,0},{0,0,1},{-1,0,0},{0,-1,0},{0,0,-1}};
        image::matrix<4,4,float> mat = image::inverse(m);
        image::vector<3,float> dir(mat.begin()+8);
        float max_cos = 0;
        for (unsigned int index = 0;index < 6;++index)
        if (dir*image::vector<3,float>(view_dirs[index]) < max_cos)
        {
            max_cos = dir*image::vector<3,float>(view_dirs[index]);
            cur_view = index;
        }
    }
    return cur_view;
}
void handleAlpha(image::rgb_color color,
                 float alpha,int blend1,int blend2)
{
    if(alpha != 1.0)
    {
        glEnable(GL_BLEND);
        glBlendFunc (BlendFunc1[blend1],
                     BlendFunc2[blend2]);
    }
    GLfloat material2[4] = { 0.0f, 0.0f, 0.0f, 0.5f};
    material2[0] = color.r/255.0;
    material2[1] = color.g/255.0;
    material2[2] = color.b/255.0;
    material2[3] = alpha*((float)color.a)/255.0;
    glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,material2);
}

void drawRegion(RegionModel& cur_region,unsigned char cur_view,
                float alpha,int blend1,int blend2)
{
    if(!cur_region.get() || cur_region.get()->tri_list.empty())
        return;
    handleAlpha(cur_region.color,alpha,blend1,blend2);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, (float*)&cur_region.get()->point_list.front());
    glNormalPointer(GL_FLOAT, 0, (float*)&cur_region.get()->normal_list.front());
    glDrawElements(GL_TRIANGLES, cur_region.getSortedIndex(cur_view).size(),
                   GL_UNSIGNED_INT,&*cur_region.getSortedIndex(cur_view).begin());
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    check_error(__FUNCTION__);
}

void my_gluLookAt(GLdouble eyex, GLdouble eyey, GLdouble eyez, GLdouble centerx,
          GLdouble centery, GLdouble centerz, GLdouble upx, GLdouble upy,
          GLdouble upz)
{
    image::vector<3,float> forward, side, up;
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
    glTranslated(-eyex, -eyey, -eyez);
}

void GLWidget::setFrustum(void)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float p[11] = {0.35f,0.4f,0.45f,0.5f,0.6f,0.8f,1.0f,1.5f,2.0f,12.0f,50.0f};
    GLfloat perspective = p[get_param("pespective")];
    GLfloat zNear = 1.0f;
    GLfloat zFar = 1000.0f;
    GLfloat aspect = float(view_mode == view_mode_type::two ? cur_width/2:cur_width)/float(cur_height);
    GLfloat fH = 0.25f;
    GLfloat fW = fH * aspect;
    glFrustum( -fW, fW, -fH, fH, zNear*perspective, zFar*perspective);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    my_gluLookAt(0,0,-200.0*perspective,0,0,0,0,-1.0,0);
}

void GLWidget::initializeGL()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
    glBlendFunc (GL_DST_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    tracts = glGenLists(1);
    tract_alpha = -1; // ensure that make_track is called
    odf_position = 255;//ensure ODFs is renderred
    check_error(__FUNCTION__);
}
void GLWidget::paintGL()
{
    glDrawBuffer(GL_BACK);
    int color = get_param("bkg_color");
    qglClearColor(QColor((float)((color & 0x00FF0000) >> 16),
                  (float)((color & 0x0000FF00) >> 8),
                  (float)(color & 0x000000FF)));
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    check_error("begin");

    setFrustum();
    check_error("basic");
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
        glGetError(); //slience the multisample error

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
            if(!get_param("stereoscopy"))
                renderLR();
            else
            {
                glDrawBuffer(GL_BACK_RIGHT);
                renderLR();
                glDrawBuffer(GL_BACK_LEFT);
                renderLR();
                glDrawBuffer(GL_BACK);
            }
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
            image::matrix<4,4,float> T(transformation_matrix);
            // add a rotation to the transofrmation matrix
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
}


void GLWidget::renderLR()
{
    std::shared_ptr<SliceModel> current_slice = cur_tracking_window.current_slice;


    if (cur_tracking_window.handle->has_odfs() &&
        get_param("show_odf"))
    {
        float fa_threshold = cur_tracking_window["fa_threshold"].toFloat();
        if(odf_position != get_param("odf_position") ||
           odf_skip != get_param("odf_skip") ||
           odf_scale != get_param("odf_scale") ||
           (get_param("odf_position") <=1 && (odf_dim != cur_tracking_window.cur_dim ||
                                               odf_slide_pos != current_slice->slice_pos[cur_tracking_window.cur_dim])))
        {
            odf_position = get_param("odf_position");
            odf_skip = get_param("odf_skip");
            odf_scale = get_param("odf_scale");
            odf_dim = cur_tracking_window.cur_dim;
            odf_slide_pos = current_slice->slice_pos[cur_tracking_window.cur_dim];
            odf_points.clear();
        }

        std::shared_ptr<fib_data> handle = cur_tracking_window.handle;
        unsigned char skip_mask_set[3] = {0,1,3};
        unsigned char mask = skip_mask_set[odf_skip];
        auto& slice = cur_tracking_window.slices[0];
        auto& geo = cur_tracking_window.handle->dim;
        if(odf_points.empty())
        switch(odf_position) // along slide
        {

        case 0:
            {
                image::geometry<2> geo2(slice->geometry[odf_dim==0?1:0],
                                       slice->geometry[odf_dim==2?1:2]);
                for(image::pixel_index<2> index(geo2);index < geo2.size();++index)
                {
                    if((index[0] & mask) | (index[1] & mask))
                        continue;
                    int x,y,z;
                    if (!slice->to3DSpace(cur_tracking_window.cur_dim,index[0],index[1],x,y,z))
                        continue;
                    image::pixel_index<3> pos(x,y,z,geo);
                    if (handle->dir.get_fa(pos.index(),0) <= fa_threshold)
                        continue;
                    add_odf(pos);
                }
            }
            break;
        case 1: // intersection
            add_odf(image::pixel_index<3>(slice->slice_pos[0],slice->slice_pos[1],slice->slice_pos[2],
                                          geo));
            break;
        case 2: //all
            for(image::pixel_index<3> index(geo);index < geo.size();++index)
            {
                if(((index[0] & mask) | (index[1] & mask) | (index[2] & mask)) ||
                   handle->dir.get_fa(index.index(),0) <= fa_threshold)
                    continue;
                add_odf(index);
            }
            break;
        }
        if(odf_colors.empty())
        {
            for (unsigned int index = 0; index <
                 cur_tracking_window.odf_size; ++index)
            {
                odf_colors.push_back(std::abs(handle->dir.odf_table[index][0]));
                odf_colors.push_back(std::abs(handle->dir.odf_table[index][1]));
                odf_colors.push_back(std::abs(handle->dir.odf_table[index][2]));
            }
        }

        glEnable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);
        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        unsigned int num_odf = odf_points.size()/cur_tracking_window.odf_size;
        unsigned int face_size = cur_tracking_window.odf_face_size*3;
        for(unsigned int index = 0,base_index = 0;index < num_odf;
            ++index,base_index += cur_tracking_window.odf_size)
        {
            glVertexPointer(3, GL_FLOAT, 0, (float*)&odf_points[base_index]);
            glColorPointer(3, GL_FLOAT, 0, (float*)&odf_colors.front());
            glDrawElements(GL_TRIANGLES, face_size,
                           GL_UNSIGNED_SHORT,handle->dir.odf_faces[0].begin());
        }
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glPopMatrix();
        glDisable(GL_COLOR_MATERIAL);

    }

    if (tracts && get_param("show_tract"))
    {
        glEnable(GL_COLOR_MATERIAL);
        if(get_param("tract_style") != 1)// 1 = tube
            glDisable(GL_LIGHTING);
        else
            setupLight((float)(get_param("tract_light_ambient"))/10.0,
                   (float)(get_param("tract_light_diffuse"))/10.0,
                   (float)(get_param("tract_light_specular"))/10.0,
                   (float)(get_param("tract_light_dir"))*3.1415926*2.0/10.0,
                   (float)(get_param("tract_light_shading"))*3.1415926/20.0,
                   get_param("tract_light_option"));

        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());
        setupMaterial((float)(get_param("tract_emission"))/10.0,
                      (float)(get_param("tract_specular"))/10.0,
                      get_param("tract_shininess")*10);

        if(get_param("tract_color_style") != tract_color_style)
        {
            if(get_param("tract_color_style") > 1)
                cur_tracking_window.color_bar->show();
            else
                cur_tracking_window.color_bar->hide();
        }


        bool changed =
           check_change("tract_alpha",tract_alpha) ||
           check_change("tract_alpha_style",tract_alpha_style) ||
           check_change("tract_style",tract_style) ||
           check_change("tract_color_style",tract_color_style) ||
           check_change("tube_diameter",tube_diameter) ||
           check_change("tract_tube_detail",tract_tube_detail) ||
           check_change("tract_variant_size",tract_variant_size) ||
           check_change("tract_variant_color",tract_variant_color) ||
           check_change("end_point_shift",end_point_shift);
        if(changed)
            makeTracts();

        if(get_param_float("tract_alpha") != 1.0)
        {
            glEnable(GL_BLEND);
            glBlendFunc (BlendFunc1[get_param("tract_bend1")],
                         BlendFunc2[get_param("tract_bend2")]);
            glDepthMask(tract_alpha_style);
        }
        else
        {
            glDisable(GL_BLEND);
            glDepthMask(true);
        }


        if(get_param("tract_shader"))
        {
            if(!shader.get())
            {
                QFile source1(":/data/shader_fragment.txt"),source2(":/data/shader_vertex.txt"),
                      source3(":/data/shader_fragment2.txt"),source4(":/data/shader_vertex2.txt");
                if (source1.open(QIODevice::ReadOnly | QIODevice::Text) &&
                    source2.open(QIODevice::ReadOnly | QIODevice::Text) &&
                    source3.open(QIODevice::ReadOnly | QIODevice::Text) &&
                    source4.open(QIODevice::ReadOnly | QIODevice::Text))
                {
                    QTextStream in1(&source1),in2(&source2),in3(&source3),in4(&source4);
                    shader = std::make_shared<QOpenGLShaderProgram>(this);
                    shader2 = std::make_shared<QOpenGLShaderProgram>(this);
                    if(!shader->addShaderFromSourceCode(QOpenGLShader::Fragment, in1.readAll().toStdString().c_str()))
                        std::cout << "Add shader fragment failed:" << shader->log().toStdString() << std::endl;
                    if(!shader->addShaderFromSourceCode(QOpenGLShader::Vertex, in2.readAll().toStdString().c_str()))
                        std::cout << "Add shader vertex failed:" << shader->log().toStdString() << std::endl;
                    if(!shader2->addShaderFromSourceCode(QOpenGLShader::Fragment, in3.readAll().toStdString().c_str()))
                        std::cout << "Add shader2 fragment failed:" << shader->log().toStdString() << std::endl;
                    if(!shader2->addShaderFromSourceCode(QOpenGLShader::Vertex, in4.readAll().toStdString().c_str()))
                        std::cout << "Add shader2 vertex failed:" << shader->log().toStdString() << std::endl;
                    if(!shader->link())
                        std::cout << "Shader failed to link:" << shader->log().toStdString() << std::endl;
                    if(!shader2->link())
                        std::cout << "Shader failed to link:" << shader->log().toStdString() << std::endl;

                    s_mvp = shader->uniformLocation( "mvp" );
                    s_mvp2 = shader2->uniformLocation( "mvp2" );
                    s_depthMap = shader2->uniformLocation( "depthMap" );

                }
                else
                    std::cout << "Failed to load shader." << std::endl;
            }
        }
        if(shader.get() && get_param("tract_shader"))
        {
            if(!shader->bind())
                std::cout << "Shader failed to bind:" << shader->log().toStdString() << std::endl;

            QMatrix4x4 model,projection,mvp;
            glGetFloatv(GL_MODELVIEW_MATRIX, model.data());
            glGetFloatv(GL_PROJECTION_MATRIX, projection.data());
            mvp = projection*model;

            shader->setUniformValue( s_mvp,mvp);
            //shader2->setUniformValue( s_mvp2, mvp);
            /*
            QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

            GLuint FramebufferName = 0;
            glFuncs->glGenFramebuffers(1, &FramebufferName);
            glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

            // create a framebuffer object for rendering the depth map:
            unsigned int depthMapFBO;
            glFuncs->glGenFramebuffers(1, &depthMapFBO);
            // create a 2D texture that we'll use as the framebuffer's depth buffer:
            const unsigned int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;

            unsigned int depthMap;
            glGenTextures(1, &depthMap);
            glBindTexture(GL_TEXTURE_2D, depthMap);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                         SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

            // attach it as the framebuffer's depth buffer:
            glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
            glFuncs->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
            glDrawBuffer(GL_NONE);
            glReadBuffer(GL_NONE);
            glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, 0);
            // 1. first render to depth map
            glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
            glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
            glClear(GL_DEPTH_BUFFER_BIT);
            //glCallList(tracts);
            //shader->release();
            // back to original viewport and frame
            glFuncs->glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, cur_width, cur_height);
            glBindTexture(GL_TEXTURE_2D, depthMap);
            //shader2->setUniformValue( s_depthMap, depthMap);
            //if(!shader2->bind())
            //    std::cout << "Shader failed to bind:" << shader->log().toStdString() << std::endl;
            */
        }
        glCallList(tracts);
        glPopMatrix();
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);
        glDepthMask(true);
        if(shader.get() && get_param("tract_shader"))
            shader->release();
        check_error("show_tract");
    }

    if (get_param("show_slice"))
    {
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);
        float alpha = get_param_float("slice_alpha");
        handleAlpha(image::rgb_color(0,0,0,255),
                        alpha,get_param("slice_bend1"),get_param("slice_bend2"));
        glDepthMask((alpha == 1.0));

        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());

        std::vector<image::vector<3,float> > points(4);



        bool show_slice[3];
        show_slice[0] = cur_tracking_window.ui->glSagCheck->checkState();
        show_slice[1] = cur_tracking_window.ui->glCorCheck->checkState();
        show_slice[2] = cur_tracking_window.ui->glAxiCheck->checkState();

        for(unsigned char dim = 0;dim < 3;++dim)
        {
            if(!show_slice[dim])
                continue;

            if(slice_pos[dim] != current_slice->slice_pos[dim])
            {
                if(slice_texture[dim])
                    deleteTexture(slice_texture[dim]);
                image::color_image texture;
                current_slice->get_texture(dim,texture,cur_tracking_window.v2c,
                                           cur_tracking_window.overlay_slice.get(),cur_tracking_window.overlay_v2c);
                slice_texture[dim] =
                    bindTexture(QImage((unsigned char*)&*texture.begin(),
                texture.width(),texture.height(),QImage::Format_RGB32));
                slice_pos[dim] = current_slice->slice_pos[dim];
            }

            glBindTexture(GL_TEXTURE_2D, slice_texture[dim]);
            int texparam[] = {GL_NEAREST,
                               GL_LINEAR};
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,texparam[get_param("slice_mag_filter")]);

            glBegin(GL_QUADS);
            glColor4f(1.0,1.0,1.0,std::min(alpha+0.2,1.0));

            slice_location(dim,points);
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
        glPopMatrix();
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);
        glDepthMask(true);
        check_error("show_slice");
    }


    if (get_param("show_region"))
    {
        glDisable(GL_COLOR_MATERIAL);
        setupLight((float)(get_param("region_light_ambient"))/10.0,
                   (float)(get_param("region_light_diffuse"))/10.0,
                   (float)(get_param("region_light_specular"))/10.0,
                   (float)(get_param("region_light_dir"))*3.1415926*2.0/10.0,
                   (float)(get_param("region_light_shading"))*3.1415926/20.0,
                   get_param("region_light_option"));

        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());

        setupMaterial((float)(get_param("region_emission"))/10.0,
                      (float)(get_param("region_specular"))/10.0,
                      get_param("region_shininess")*10);

        float alpha = get_param_float("region_alpha");
        unsigned char cur_view = (alpha == 1.0 ? 0 : getCurView(transformation_matrix));

        std::vector<unsigned int> region_need_update;
        for(unsigned int index = 0;index < cur_tracking_window.regionWidget->regions.size();++index)
            if(cur_tracking_window.regionWidget->item(index,0)->checkState() == Qt::Checked &&
                    cur_tracking_window.regionWidget->regions[index]->modified)
                region_need_update.push_back(index);

        int smoothed = get_param("region_mesh_smoothed");
        image::par_for(region_need_update.size(),[&](unsigned int index){
            cur_tracking_window.regionWidget->regions[region_need_update[index]]->makeMeshes(smoothed);
        });

        for(unsigned int index = 0;index < cur_tracking_window.regionWidget->regions.size();++index)
            if(cur_tracking_window.regionWidget->item(index,0)->checkState() == Qt::Checked)
            {
                drawRegion(cur_tracking_window.regionWidget->regions[index]->show_region,
                           cur_view,alpha,get_param("region_bend1"),get_param("region_bend2"));
            }
        glDisable(GL_BLEND);
        glPopMatrix();
        check_error("show_region");

    }

    if (surface.get() && get_param("show_surface"))
    {
        glDisable(GL_COLOR_MATERIAL);
        setupLight((float)(get_param("surface_light_ambient"))/10.0,
                   (float)(get_param("surface_light_diffuse"))/10.0,
                   (float)(get_param("surface_light_specular"))/10.0,
                   (float)(get_param("surface_light_dir"))*3.1415926*2.0/10.0,
                   (float)(get_param("surface_light_shading"))*3.1415926/20.0,
                   get_param("surface_light_option"));

        glPushMatrix();
        glMultMatrixf(transformation_matrix.begin());
        setupMaterial((float)(get_param("surface_emission"))/10.0,
                      (float)(get_param("surface_specular"))/10.0,
                      get_param("surface_shininess")*10);

        float alpha = get_param_float("surface_alpha");
        surface->color = (unsigned int)get_param("surface_color");
        surface->color.a = 255;
        unsigned char cur_view = (alpha == 1.0 ? 0 : getCurView(transformation_matrix));
        drawRegion(*surface.get(),cur_view,alpha,
                   get_param("surface_bend1"),
                   get_param("surface_bend2"));
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
            glColor3ub((color & 0x00FF0000) >> 16,(color & 0x0000FF00) >> 8,color & 0x000000FF);
            QFont font;
            font.setPointSize(get_param("track_label_size"));
            font.setBold(get_param("track_label_bold"));
            for (unsigned int i = 0;i < cur_tracking_window.tractWidget->rowCount();++i)
            if(cur_tracking_window.tractWidget->item(i,0)->checkState() == Qt::Checked)
            {
                TractModel* active_tract_model =
                    cur_tracking_window.tractWidget->tract_models[i];
                if (active_tract_model->get_visible_track_count() == 0)
                    continue;
                const auto& t = active_tract_model->get_tract(0);
                int pos = t.size()/6*3;
                renderText(t[pos],t[pos+1],t[pos+2],cur_tracking_window.tractWidget->item(i,0)->text(),font);
            }
        }
        if (get_param("show_region_label"))
        {
            int color = get_param("region_label_color");
            glColor3ub((color & 0x00FF0000) >> 16,(color & 0x0000FF00) >> 8,color & 0x000000FF);
            QFont font;
            font.setPointSize(get_param("region_label_size"));
            font.setBold(get_param("region_label_bold"));
            for(unsigned int i = 0;i < cur_tracking_window.regionWidget->regions.size();++i)
                if(cur_tracking_window.regionWidget->item(i,0)->checkState() == Qt::Checked &&
                   !cur_tracking_window.regionWidget->regions[i]->region.empty())
                {
                    const auto& p = cur_tracking_window.regionWidget->regions[i]->show_region.center;
                    if(p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0)
                    {
                        const auto& p2 = cur_tracking_window.regionWidget->regions[i]->show_region.center;
                        image::vector<3> p3(p2);
                        p3 /= cur_tracking_window.regionWidget->regions[i]->resolution_ratio;
                        renderText(p3[0],p3[1],p3[2],cur_tracking_window.regionWidget->item(i,0)->text(),font);
                    }
                    else
                    renderText(p[0],p[1],p[2],cur_tracking_window.regionWidget->item(i,0)->text(),font);
                }
        }
        glPopMatrix();
        glEnable(GL_LIGHTING);
        glEnable(GL_DEPTH_TEST);

    }

    if (get_param("show_axis"))
    {
        float L = get_param_float("axis_line_length");
        float L2 = L*0.75f;


        glEnable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        float p[11] = {0.35f,0.4f,0.45f,0.5f,0.6f,0.8f,1.0f,1.5f,2.0f,12.0f,50.0f};
        GLfloat perspective = p[get_param("pespective")];
        GLfloat zNear = 1.0f;
        GLfloat zFar = 1000.0f;
        GLfloat aspect = float(view_mode == view_mode_type::two ? cur_width/2:cur_width)/float(cur_height);
        GLfloat fH = 0.25f;
        GLfloat fW = fH * aspect;
        glFrustum( -fW, L2*0.01f,L2*-0.01f, fH, zNear*perspective, zFar*perspective);


        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        my_gluLookAt(0,0,-200.0f*perspective,0,0,0,0,-1.0f,0);
        glMultMatrixf(rotation_matrix.begin());
        glLineWidth (get_param_float("axis_line_thickness"));
        glBegin (GL_LINES);
        glColor3f (1.0f,0.3f,0.3f);  glVertex3f(0,0,0);  glVertex3f(L2,0,0);    // X axis is red.
        glColor3f (0.3f,1.0f,0.3f);  glVertex3f(0,0,0);  glVertex3f(0,L2,0);    // Y axis is green.
        glColor3f (0.3f,0.3f,1.0f);  glVertex3f(0,0,0);  glVertex3f(0,0,L2);    // z axis is blue.
        glEnd();
        if(get_param("show_axis_label"))
        {
            QFont font;
            font.setPointSize(get_param("axis_label_size"));
            font.setBold(get_param("axis_label_bold"));
            renderText(0,0,L,"S",font);
            glColor3f (1.0f,0.3f,0.3f);
            renderText(L,0,0,"L",font);
            glColor3f (0.3f,1.0f,0.3f);
            renderText(0,L,0,"P",font);
        }
        glEnable(GL_DEPTH_TEST);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        check_error("axis");
    }
}


void GLWidget::add_odf(image::pixel_index<3> pos)
{
    std::shared_ptr<fib_data> handle = cur_tracking_window.handle;
    const float* odf_buffer =
            handle->get_odf_data(pos.index());
    if(!odf_buffer)
        return;
    unsigned int odf_dim = cur_tracking_window.odf_size;
    unsigned int half_odf = odf_dim >> 1;
    odf_points.resize(odf_points.size()+odf_dim);
    std::vector<image::vector<3,float> >::iterator iter = odf_points.end()-odf_dim;
    std::vector<image::vector<3,float> >::iterator end = odf_points.end();
    std::fill(iter,end,pos);

    float odf_min = *std::min_element(odf_buffer,odf_buffer+half_odf);
    float max_fa = *std::max_element(cur_tracking_window.handle->dir.fa[0],cur_tracking_window.handle->dir.fa[0] + cur_tracking_window.handle->dim.size());
    if(max_fa == 0.0)
        max_fa = 1.0;
    float scaling = odf_scale/max_fa;
    if(!get_param("odf_min_max"))
    {
        scaling = 0.5*odf_scale/(max_fa+odf_min);
        odf_min = 0;
    }
    // smooth the odf a bit

    std::vector<float> new_odf_buffer;
    if(get_param("odf_smoothing"))
    {
        new_odf_buffer.resize(half_odf);
        std::copy(odf_buffer,odf_buffer+half_odf,new_odf_buffer.begin());
        std::vector<image::vector<3,unsigned short> >& odf_faces =
                handle->dir.odf_faces;
        for(int index = 0;index < odf_faces.size();++index)
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


    for(unsigned int index = 0;index < half_odf;++index,++iter)
    {
        image::vector<3,float> displacement(handle->dir.odf_table[index]);
        displacement *= (odf_buffer[index]-odf_min)*scaling;
        *(iter) += displacement;
        *(iter+half_odf) -= displacement;
    }
}
void myglColor(const image::vector<3,float>& color,float alpha)
{
    if(alpha == 1.0)
        glColor3fv(color.begin());
    else
        glColor4f(color[0],color[1],color[2],alpha);
}

void GLWidget::makeTracts(void)
{
    if(!tracts)
        return;
    makeCurrent();
    glDeleteLists(tracts, 1);
    glNewList(tracts, GL_COMPILE);
    float alpha = (tract_alpha_style == 0)? tract_alpha/2.0f:tract_alpha;
    const float detail_option[] = {1.0f,0.5f,0.25f,0.0f,0.0f};
    bool show_end_points = tract_style == 2;
    const float variant_prob_option[] = {0.1f,0.2f,0.4f,0.95f,2.0f};
    float variant_prob = variant_prob_option[tract_tube_detail];
    float tube_detail = tube_diameter*detail_option[tract_tube_detail]*4.0f;
    float skip_rate = 1.0;
    std::vector<float> mean_fa;
    unsigned int mean_fa_index = 0;
    unsigned int track_num_index = cur_tracking_window.handle->get_name_index(cur_tracking_window.color_bar->get_tract_color_name().toStdString());
    // show tract by index value
    if (tract_color_style > 1)
    {
        if(tract_color_style == 3 || tract_color_style == 5)// mean value
        {
            for (unsigned int active_tract_index = 0;
                    active_tract_index < cur_tracking_window.tractWidget->rowCount();
                    ++active_tract_index)
            {
                if(cur_tracking_window.tractWidget->item(active_tract_index,0)->checkState() != Qt::Checked)
                    continue;
                TractModel* active_tract_model =
                    cur_tracking_window.tractWidget->tract_models[active_tract_index];
                if (active_tract_model->get_visible_track_count() == 0)
                    continue;
                unsigned int tracks_count = active_tract_model->get_visible_track_count();
                for (unsigned int data_index = 0; data_index < tracks_count; ++data_index)
                {
                    unsigned int vertex_count =
                            active_tract_model->get_tract_length(data_index)/3;
                    if (vertex_count <= 1)
                        continue;

                    std::vector<float> fa_values;
                    active_tract_model->get_tract_data(data_index,track_num_index,fa_values);
                    if(tract_color_style == 3)
                    {
                        float sum = std::accumulate(fa_values.begin(),fa_values.end(),0.0f);
                        sum /= (float)fa_values.size();
                        mean_fa.push_back(sum);
                    }
                    if(tract_color_style == 5)
                        mean_fa.push_back(*std::max_element(fa_values.begin(),fa_values.end()));
                }
            }
        }
    }

    std::vector<image::vector<3,float> > points(8),previous_points(8),
                                      normals(8),previous_normals(8);
    image::rgb_color paint_color;
    image::vector<3,float> paint_color_f;
    std::vector<float> color;

    unsigned int visible_tracts = get_param("tract_visible_tract");
    {
        unsigned int total_tracts = 0;
        for (unsigned int active_tract_index = 0;
                active_tract_index < cur_tracking_window.tractWidget->rowCount();
                ++active_tract_index)
        {
            if(cur_tracking_window.tractWidget->item(active_tract_index,0)->checkState() != Qt::Checked)
                continue;
            TractModel* active_tract_model =
                cur_tracking_window.tractWidget->tract_models[active_tract_index];
            if (active_tract_model->get_visible_track_count() == 0)
                continue;

            total_tracts += active_tract_model->get_visible_track_count();
        }
        if(total_tracts != 0)
            skip_rate = (float)visible_tracts/(float)total_tracts;
    }
    image::uniform_dist<float> uniform_gen(0.0f,1.0f),random_size(-0.5f,0.5f),random_color(-0.05f,0.05f);
    {
        for (unsigned int active_tract_index = 0;
                active_tract_index < cur_tracking_window.tractWidget->rowCount();
                ++active_tract_index)
        {
            if(cur_tracking_window.tractWidget->item(active_tract_index,0)->checkState() != Qt::Checked)
                continue;
            TractModel* active_tract_model =
                cur_tracking_window.tractWidget->tract_models[active_tract_index];
            if (active_tract_model->get_visible_track_count() == 0)
                continue;

            unsigned int tracks_count = active_tract_model->get_visible_track_count();
            for (unsigned int data_index = 0; data_index < tracks_count; ++data_index)
            {
                unsigned int vertex_count =
                        active_tract_model->get_tract_length(data_index)/3;
                if (vertex_count <= 1)
                    continue;

                if(skip_rate < 1.0 && uniform_gen() > skip_rate)
                {
                    mean_fa_index++;
                    continue;
                }

                const float* data_iter = &*(active_tract_model->get_tract(data_index).begin());

                switch(tract_color_style)
                {
                case 1:
                    paint_color = active_tract_model->get_tract_color(data_index);
                    paint_color_f = image::vector<3,float>(paint_color.r,paint_color.g,paint_color.b);
                    paint_color_f /= 255.0;
                    break;
                case 2:// local
                    active_tract_model->get_tract_data(data_index,track_num_index,color);
                    break;
                case 3:// mean
                case 5:// max
                    paint_color_f = cur_tracking_window.color_bar->get_color(mean_fa[mean_fa_index++]);
                    break;
                }
                image::vector<3,float> last_pos(data_iter),pos,
                    vec_a(1,0,0),vec_b(0,1,0),
                    vec_n,prev_vec_n,vec_ab,vec_ba,cur_color,previous_color;

                glBegin((tract_style) ? GL_TRIANGLE_STRIP : GL_LINE_STRIP);
                for (unsigned int j = 0, index = 0; index < vertex_count;j += 3, data_iter += 3, ++index)
                {
                    pos[0] = data_iter[0];
                    pos[1] = data_iter[1];
                    pos[2] = data_iter[2];
                    if (index + 1 < vertex_count)
                    {
                        vec_n[0] = data_iter[3] - data_iter[0];
                        vec_n[1] = data_iter[4] - data_iter[1];
                        vec_n[2] = data_iter[5] - data_iter[2];
                        vec_n.normalize();
                    }

                    switch(tract_color_style)
                    {
                    case 0://directional
                        cur_color[0] = std::fabs(vec_n[0]);
                        cur_color[1] = std::fabs(vec_n[1]);
                        cur_color[2] = std::fabs(vec_n[2]);
                        break;
                    case 1://manual assigned
                    case 3://mean anisotropy
                    case 4://mean directional
                    case 5://max anisotropy
                        cur_color = paint_color_f;
                        break;
                    case 2://local anisotropy
                        if(index < color.size())
                            cur_color = cur_tracking_window.color_bar->get_color(color[index]);
                        break;
                    }

                    if(tract_variant_color)
                    {
                        cur_color[0] += random_color();
                        cur_color[1] += random_color();
                        cur_color[2] += random_color();
                    }

                    // skip straight line!
                    if(tract_style)
                    {
                    if (index != 0 && index+1 != vertex_count)
                    {
                        image::vector<3,float> displacement(data_iter+3);
                        displacement -= last_pos;
                        displacement -= prev_vec_n*(prev_vec_n*displacement);
                        if (displacement.length() < tube_detail)
                            continue;
                    }
\
                    if (index == 0 && std::fabs(vec_a*vec_n) > 0.5)
                        std::swap(vec_a,vec_b);

                    vec_b = vec_a.cross_product(vec_n);
                    vec_a = vec_n.cross_product(vec_b);
                    vec_a.normalize();
                    vec_b.normalize();
                    vec_ba = vec_ab = vec_a;
                    vec_ab += vec_b;
                    vec_ba -= vec_b;
                    vec_ab.normalize();
                    vec_ba.normalize();
                    // get normals
                    {
                        normals[0] = vec_a;
                        normals[1] = vec_ab;
                        normals[2] = vec_b;
                        normals[3] = -vec_ba;
                        normals[4] = -vec_a;
                        normals[5] = -vec_ab;
                        normals[6] = -vec_b;
                        normals[7] = vec_ba;
                    }
                    if(tract_variant_size && uniform_gen() > variant_prob)
                    {
                        vec_ab += random_size();
                        vec_ba += random_size();
                        vec_a += random_size();
                        vec_b += random_size();
                    }
                    vec_ab *= tube_diameter;
                    vec_ba *= tube_diameter;
                    vec_a *= tube_diameter;
                    vec_b *= tube_diameter;

                    // add point
                    {
                        std::fill(points.begin(),points.end(),pos);
                        points[0] += vec_a;
                        points[1] += vec_ab;
                        points[2] += vec_b;
                        points[3] -= vec_ba;
                        points[4] -= vec_a;
                        points[5] -= vec_ab;
                        points[6] -= vec_b;
                        points[7] += vec_ba;
                    }
                    // add end
                    static const unsigned char end_sequence[8] = {4,3,5,2,6,1,7,0};
                    if (index == 0)
                    {
                        /*
                        glColor3fv(cur_color.begin());
                        glNormal3f(-vec_n[0],-vec_n[1],-vec_n[2]);
                        for (unsigned int k = 0;k < 8;++k)
                            glVertex3fv(points[end_sequence[k]].begin());
                        if(show_end_points)
                            glEnd();*/
                        if(show_end_points)
                        {
                            myglColor(cur_color,alpha);
                            glNormal3f(-vec_n[0],-vec_n[1],-vec_n[2]);
                            image::vector<3,float> shift(vec_n);
                            shift *= -(int)end_point_shift;
                            for (unsigned int k = 0;k < 8;++k)
                            {
                                image::vector<3,float> cur_point = points[end_sequence[k]];
                                cur_point += shift;
                                glVertex3fv(cur_point.begin());
                            }
                            glEnd();
                        }
                        else
                        {
                            myglColor(cur_color,alpha);
                            glNormal3f(-vec_n[0],-vec_n[1],-vec_n[2]);
                            for (unsigned int k = 0;k < 8;++k)
                                glVertex3fv(points[end_sequence[k]].begin());
                        }
                    }
                    else
                    // add tube
                    {

                        if(!show_end_points)
                        {
                            myglColor(cur_color,alpha);
                            glNormal3fv(normals[0].begin());
                            glVertex3fv(points[0].begin());
                            for (unsigned int k = 1;k < 8;++k)
                            {
                               myglColor(previous_color,alpha);
                               glNormal3fv(previous_normals[k].begin());
                               glVertex3fv(previous_points[k].begin());

                               myglColor(cur_color,alpha);
                               glNormal3fv(normals[k].begin());
                               glVertex3fv(points[k].begin());
                            }
                            myglColor(cur_color,alpha);
                            glNormal3fv(normals[0].begin());
                            glVertex3fv(points[0].begin());
                        }
                        if(index +1 == vertex_count)
                        {
                            if(show_end_points)
                            {
                                glBegin((tract_style) ? GL_TRIANGLE_STRIP : GL_LINE_STRIP);
                                myglColor(cur_color,alpha);
                                glNormal3fv(vec_n.begin());
                                image::vector<3,float> shift(vec_n);
                                shift *= (int)end_point_shift;
                                for (int k = 7;k >= 0;--k)
                                {
                                    image::vector<3,float> cur_point = points[end_sequence[k]];
                                    cur_point += shift;
                                    glVertex3fv(cur_point.begin());
                                }
                            }
                            else
                            {
                                myglColor(cur_color,alpha);
                                glNormal3fv(vec_n.begin());
                                for (int k = 7;k >= 0;--k)
                                    glVertex3fv(points[end_sequence[k]].begin());
                            }
                        }

                    }

                    previous_points.swap(points);
                    previous_normals.swap(normals);
                    previous_color = cur_color;
                    prev_vec_n = vec_n;
                    last_pos = pos;
                    }
                    else
                    {
                        myglColor(cur_color,alpha);
                        glVertex3fv(pos.begin());
                    }
                }
                glEnd();
            }
        }
    }


    glEndList();

    check_error(__FUNCTION__);
}
void GLWidget::resizeGL(int width_, int height_)
{
    cur_width = width_;
    cur_height = height_;
}
void GLWidget::scale_by(float scalefactor)
{
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
    updateGL();
}

void GLWidget::wheelEvent ( QWheelEvent * event )
{
    edit_right = (view_mode != view_mode_type::single && (event->x() > cur_width / 2));
    double scalefactor = event->delta();
    scalefactor /= 1200.0;
    scalefactor = 1.0+scalefactor;
    scale_by(scalefactor);
    event->ignore();

}

void GLWidget::slice_location(unsigned char dim,std::vector<image::vector<3,float> >& points)
{
    cur_tracking_window.current_slice->get_slice_positions(dim,points);
}

void GLWidget::get_view_dir(QPoint p,image::vector<3,float>& dir)
{
    image::matrix<4,4,float> m;
    float v[3];
    glGetFloatv(GL_PROJECTION_MATRIX,m.begin());
    // Compute the vector of the pick ray in screen space
    v[0] = (( 2.0f * ((float)p.x() * devicePixelRatio())/float(view_mode == view_mode_type::two ? cur_width/2:cur_width)) - 1 ) / m[0];
    v[1] = -(( 2.0f * ((float)p.y() * devicePixelRatio())/((float)cur_height)) - 1 ) / m[5];
    v[2] = -1.0f;
    // Transform the screen space pick ray into 3D space
    dir[0] = v[0]*mat[0] + v[1]*mat[4] + v[2]*mat[8];
    dir[1] = v[0]*mat[1] + v[1]*mat[5] + v[2]*mat[9];
    dir[2] = v[0]*mat[2] + v[1]*mat[6] + v[2]*mat[10];
    dir.normalize();
}

float GLWidget::get_slice_projection_point(unsigned char dim,
                                const image::vector<3,float>& pos,
                                const image::vector<3,float>& dir,
                                float& dx,float& dy)
{
    std::vector<image::vector<3,float> > slice_points(4);
    slice_location(dim,slice_points);
    image::vector<3,float> pos_offset(pos),v1(slice_points[1]),v2(slice_points[2]),v3(dir);
    pos_offset -= slice_points[0];
    v1 -= slice_points[0];
    v2 -= slice_points[0];
    image::matrix<3,3,float> m;
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

image::vector<3,float> get_norm(const std::vector<image::vector<3,float> >& slice_points)
{
    image::vector<3,float> v1(slice_points[1]),v2(slice_points[2]),norm;
    v1 -= slice_points[0];
    v2 -= slice_points[0];
    norm = v1.cross_product(v2);
    norm.normalize();
    return norm;
}
void GLWidget::select_object(void)
{
    object_selected = false;
    if(cur_tracking_window.regionWidget->regions.empty())
        return;
    // select object
    for(object_distance = 0;object_distance < 5000 && !object_selected;object_distance += 1.0)
    {
    image::vector<3,float> cur_pos(dir1);
    cur_pos *= object_distance;
    cur_pos += pos;
    image::vector<3,short> voxel(cur_pos);
    if(!cur_tracking_window.handle->dim.is_valid(voxel))
        continue;
    for(int index = 0;index < cur_tracking_window.regionWidget->regions.size();++index)
        if(cur_tracking_window.regionWidget->regions[index]->has_point(voxel) &&
           cur_tracking_window.regionWidget->item(index,0)->checkState() == Qt::Checked)
        {
            selected_index = index;
            object_selected = true;
            break;
        }
    }
}

void GLWidget::select_slice(void)
{
    bool show_slice[3];
    show_slice[0] = cur_tracking_window.ui->glSagCheck->checkState();
    show_slice[1] = cur_tracking_window.ui->glCorCheck->checkState();
    show_slice[2] = cur_tracking_window.ui->glAxiCheck->checkState();
    // select slice
    slice_selected = false;
    {
        // now check whether the slices are selected
        slice_distance = std::numeric_limits<float>::max();
        for(unsigned char dim = 0;dim < 3;++dim)
        {
            if(!show_slice[dim])
                continue;
            float d = get_slice_projection_point(dim,pos,dir1,slice_dx,slice_dy);
            if(slice_dx > 0.0 && slice_dy > 0.0 &&
               slice_dx < 1.0 && slice_dy < 1.0 &&
                    d > 0 && slice_distance > d)
            {
                moving_at_slice_index = dim;
                slice_distance = d;
                slice_selected = true;
            }
        }
    }
}
void GLWidget::get_pos(void)
{
    //glMultMatrixf(transformation_matrix);
    glGetFloatv(GL_MODELVIEW_MATRIX,mat.begin());
    image::matrix<4,4,float> view = (edit_right) ? transformation_matrix2*mat : transformation_matrix*mat;
    mat = image::inverse(view);
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
    select_object();
    select_slice();
    if(!object_selected && !slice_selected)
        return;
    // if only slice is selected or slice is at the front, then move slice
    if(!object_selected || object_distance > slice_distance)
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
        return;
    }

    if(object_selected)
    {
        cur_tracking_window.regionWidget->item(selected_index,0)->setCheckState(Qt::CheckState());
        emit region_edited();
    }
}
bool GLWidget::get_mouse_pos(QMouseEvent *event,image::vector<3,float>& position)
{
    QPoint cur_pos = event->pos();
    if(edit_right)
        cur_pos.setX(cur_pos.x() - cur_width / 2);

    get_pos();
    image::vector<3,float> cur_dir;
    get_view_dir(cur_pos,cur_dir);

    bool show_slice[3];
    show_slice[0] = cur_tracking_window.ui->glSagCheck->checkState();
    show_slice[1] = cur_tracking_window.ui->glCorCheck->checkState();
    show_slice[2] = cur_tracking_window.ui->glAxiCheck->checkState();
    // select slice
    slice_selected = false;
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
            std::vector<image::vector<3,float> > points(4);
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
    if(view_mode == view_mode_type::stereo)
        p.setX(p.x()*2);
    return p;
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    setFocus();// for key stroke to work
    makeCurrent();
    edit_right = (view_mode != view_mode_type::single && (event->x() > cur_width / 2));
    lastPos = convert_pos(event);
    if(editing_option != none)
        get_pos();
    if(editing_option == selecting)
    {
        dirs.clear();
        last_select_point = lastPos;
        dirs.push_back(image::vector<3,float>());
        get_view_dir(last_select_point,dirs.back());
    }
    if(editing_option == moving)
    {
        get_view_dir(lastPos,dir1);
        dir1.normalize();

        // nothing selected
        select_object();
        select_slice();

        if(!object_selected && !slice_selected)
        {
            editing_option = none;
            setCursor(Qt::ArrowCursor);
            return;
        }
        // if only slice is selected or slice is at the front, then move slice
        if(!object_selected || object_distance > slice_distance)
        {
            editing_option = dragging;
            return;
        }
        cur_tracking_window.regionWidget->selectRow(selected_index);
        // determine the moving direction of the region
        float angle[3] = {0,0,0};
        bool show_slice[3];
        show_slice[0] = cur_tracking_window.ui->glSagCheck->checkState();
        show_slice[1] = cur_tracking_window.ui->glCorCheck->checkState();
        show_slice[2] = cur_tracking_window.ui->glAxiCheck->checkState();

        for(unsigned char dim = 0;dim < 3;++dim)
        {
            std::vector<image::vector<3,float> > points(4);
            slice_location(dim,points);
            angle[dim] = std::fabs(dir1*get_norm(points)) + (show_slice[dim] ? 1:0);
        }
        moving_at_slice_index = std::max_element(angle,angle+3)-angle;
        if(get_slice_projection_point(moving_at_slice_index,pos,dir1,slice_dx,slice_dy) == 0.0)
        {
            editing_option = none;
            setCursor(Qt::ArrowCursor);
            return;
        }
        accumulated_dis = image::zero<float>();
    }
}
void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    makeCurrent();
    if(editing_option == none)
        return;
    editing_option = none;
    setCursor(Qt::ArrowCursor);
    if(editing_option == moving || editing_option == dragging)
        return;

    last_select_point = convert_pos(event);
    dirs.push_back(image::vector<3,float>());
    get_view_dir(last_select_point,dirs.back());

    angular_selection = event->button() == Qt::RightButton;
    emit edited();
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
    updateGL();
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
    QPoint cur_pos = convert_pos(event);

    makeCurrent();
    if(editing_option == selecting)
    {
        if(event->modifiers() & Qt::ShiftModifier)
        {
            QPoint dis(cur_pos);
            dis -= last_select_point;
            if(dis.manhattanLength() < 20)
                return;
            last_select_point = cur_pos;
            dirs.push_back(image::vector<3,float>());
            get_view_dir(last_select_point,dirs.back());
        }
        return;
    }

    if(editing_option == moving)
    {

        std::vector<image::vector<3,float> > points(4);
        slice_location(moving_at_slice_index,points);
        get_view_dir(cur_pos,dir2);
        float dx,dy;
        if(get_slice_projection_point(moving_at_slice_index,pos,dir2,dx,dy) == 0.0)
            return;
        image::vector<3,float> v1(points[1]),v2(points[2]),dis;
        v1 -= points[0];
        v2 -= points[0];
        dis = v1*(dx-slice_dx)+v2*(dy-slice_dy);
        dis -= accumulated_dis;
        dis.round();
        if(dis[0] != 0 || dis[1] != 0 || dis[2] != 0)
        {
            cur_tracking_window.regionWidget->regions[cur_tracking_window.regionWidget->currentRow()]->shift(dis);
            accumulated_dis += dis;
            emit region_edited();
        }
        return;
    }
    // move slice
    if(editing_option == dragging)
    {

        std::vector<image::vector<3,float> > points(4);
        slice_location(moving_at_slice_index,points);
        get_view_dir(cur_pos,dir2);
        float move_dis = (dir2-dir1)*get_norm(points);
        move_dis *= slice_distance;
        if(std::fabs(move_dis) < 1.0)
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


    float dx = cur_pos.x() - lastPos.x();
    float dy = cur_pos.y() - lastPos.y();
    if(event->modifiers() & Qt::ControlModifier)
    {
        dx /= 16.0;
        dy /= 16.0;
    }

    glPushMatrix();
    glLoadIdentity();
    // left button down
    if (event->buttons() & Qt::LeftButton)
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
    // right button or middle button down
    {
        if (event->buttons() & Qt::RightButton)
        {
            double scalefactor = (-dx-dy+100.0)/100.0;
            glScaled(scalefactor,scalefactor,scalefactor);
        }
        else
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
    updateGL();
    lastPos = cur_pos;
}

void GLWidget::saveCamera(void)
{
    QString filename = cur_tracking_window.
            get_save_file_name("Save Translocation Matrix","camera.txt","Text files (*.txt);;All files (*)");
    if(filename.isEmpty())
        return;
    std::ofstream out(filename.toLocal8Bit().begin());
    for(int row = 0,index = 0;row < 4;++row)
    {
        for(int col = 0;col < 4;++col,++index)
            out << transformation_matrix[index] << " ";
        out << std::endl;
    }
}

void GLWidget::loadCamera(void)
{
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Open Translocation Matrix",QFileInfo(cur_tracking_window.windowTitle()).absolutePath(),"Text files (*.txt);;All files (*)");
    std::ifstream in(filename.toLocal8Bit().begin());
    if(filename.isEmpty() || !in)
        return;
    std::vector<float> data;
    std::copy(std::istream_iterator<float>(in),
              std::istream_iterator<float>(),std::back_inserter(data));
    data.resize(16);
    std::copy(data.begin(),data.end(),transformation_matrix.begin());
    updateGL();
}
void GLWidget::addSurface(void)
{
    float threshold = image::segmentation::otsu_threshold(cur_tracking_window.current_slice->get_source());
    bool ok;
    threshold = QInputDialog::getDouble(this,
        "DSI Studio","Threshold:", threshold,
        *std::min_element(cur_tracking_window.current_slice->get_source().begin(),cur_tracking_window.current_slice->get_source().end()),
        *std::max_element(cur_tracking_window.current_slice->get_source().begin(),cur_tracking_window.current_slice->get_source().end()),
        4, &ok);
    if (!ok)
        return;
    QAction* pAction = qobject_cast<QAction*>(sender());
    int id = 0;
    if(pAction)
    {
        if(pAction->text() == "Full")
            id = 0;
        if(pAction->text() == "Right")
            id = 1;
        if(pAction->text() == "Left")
            id = 2;
        if(pAction->text() == "Lower")
            id = 3;
        if(pAction->text() == "Upper")
            id = 4;
        if(pAction->text() == "Anterior")
            id = 5;
        if(pAction->text() == "Posterior")
            id = 6;
    }
    command("add_surface",QString::number(id),QString::number(threshold));
}

void GLWidget::copyToClipboard(void)
{
    paintGL();
    QApplication::clipboard()->setImage(grabFrameBuffer());
}



void GLWidget::get3View(QImage& I,unsigned int type)
{
    makeCurrent();
    set_view_flip = false;
    set_view(0);
    paintGL();
    QImage image0 = grabFrameBuffer();
    set_view_flip = true;
    set_view(0);
    paintGL();
    QImage image00 = grabFrameBuffer();
    set_view_flip = true;
    set_view(1);
    paintGL();
    QImage image1 = grabFrameBuffer();
    set_view_flip = true;
    set_view(2);
    paintGL();
    QImage image2 = grabFrameBuffer();
    QImage image3 = cur_tracking_window.scene.view_image.scaledToWidth(image0.width()).convertToFormat(QImage::Format_RGB32);
    if(type == 0)
    {
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
        QImage all(image0.width()*4,image0.height(),QImage::Format_RGB32);
        QPainter painter(&all);
        painter.drawImage(0,0,image0);
        painter.drawImage(image0.width(),0,image00);
        painter.drawImage(image0.width()*2,0,image1);
        painter.drawImage(image0.width()*3,((int)image0.height()-(int)image2.height())/2,image2);
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

bool GLWidget::command(QString cmd,QString param,QString param2)
{
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
            paintGL();
        }
        return true;
    }
    if(cmd == "set_view")
    {
        if(param.isEmpty())
            return true;
        makeCurrent();
        set_view(param.toInt());
        paintGL();
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
        paintGL();
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
        paintGL();
        return true;
    }
    if(cmd == "add_surface")
    {
        float threshold = (param2.isEmpty()) ? image::segmentation::otsu_threshold(cur_tracking_window.current_slice->get_source()):param2.toFloat();
        {
            surface.reset(new RegionModel);
            image::basic_image<float, 3> crop_image(cur_tracking_window.current_slice->get_source());
            if(!param.isEmpty())
            switch(param.toInt())
            {
            case 1: //right hemi
                for(unsigned int index = 0;index < crop_image.size();index += crop_image.width())
                {
                    std::fill(crop_image.begin()+index+cur_tracking_window.current_slice->slice_pos[0],
                              crop_image.begin()+index+crop_image.width(),crop_image[0]);
                }
                break;
            case 2: // left hemi
                for(unsigned int index = 0;index < crop_image.size();index += crop_image.width())
                {
                    std::fill(crop_image.begin()+index,
                              crop_image.begin()+index+cur_tracking_window.current_slice->slice_pos[0],crop_image[0]);
                }
                break;
            case 3: //lower
                std::fill(crop_image.begin()+cur_tracking_window.current_slice->slice_pos[2]*crop_image.plane_size(),
                          crop_image.end(),crop_image[0]);
                break;
            case 4: // higher
                std::fill(crop_image.begin(),
                          crop_image.begin()+cur_tracking_window.current_slice->slice_pos[2]*crop_image.plane_size(),crop_image[0]);
                break;
            case 5: //anterior
                for(unsigned int index = 0;index < crop_image.size();index += crop_image.plane_size())
                {
                    std::fill(crop_image.begin()+index+cur_tracking_window.current_slice->slice_pos[1]*crop_image.width(),
                              crop_image.begin()+index+crop_image.plane_size(),crop_image[0]);
                }
                break;
            case 6: // posterior
                for(unsigned int index = 0;index < crop_image.size();index += crop_image.plane_size())
                {
                    std::fill(crop_image.begin()+index,
                              crop_image.begin()+index+cur_tracking_window.current_slice->slice_pos[1]*crop_image.width(),crop_image[0]);
                }
                break;
            }
            switch(get_param("surface_mesh_smoothed"))
            {
            case 1:
                image::filter::gaussian(crop_image);
                break;
            case 2:
                image::filter::gaussian(crop_image);
                image::filter::gaussian(crop_image);
                break;
            }
            if(!surface->load(crop_image,threshold))
            {
                surface.reset(0);
                return true;
            }
        }

        if(!cur_tracking_window.current_slice->is_diffusion_space)
        for(unsigned int index = 0;index < surface->get()->point_list.size();++index)
        {
            surface->get()->point_list[index].to(cur_tracking_window.current_slice->transform);
            surface->get()->point_list[index] += 0.5;
        }
        paintGL();
        return true;
    }
    if(cmd == "save_image")
    {
        if(param.isEmpty())
            param = QFileInfo(cur_tracking_window.windowTitle()).fileName()+".image.jpg";
        if(!param2.isEmpty())
        {
            std::istringstream in(param2.toLocal8Bit().begin());
            int w = 0;
            int h = 0;
            int ow = width(),oh = height();
            in >> w >> h;
            resize(w,h);
            updateGL();
            grabFrameBuffer().save(param);
            resize(ow,oh);
        }
        else
            grabFrameBuffer().save(param);
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
        begin_prog("save video");
        float angle = (param2.isEmpty()) ? 1 : param2.toFloat();
        int ow = width(),oh = height();
        image::io::avi avi;
        #ifndef __APPLE__
            resize(1980,1080);
        #endif
        for(float index = 0;check_prog(index,360);index += angle)
        {
            rotate_angle(angle,0,1.0,0.0);
            QBuffer buffer;
            QImageWriter writer(&buffer, "JPG");
            updateGL();
            QImage I = grabFrameBuffer();
            writer.write(I);
            if(index == 0.0)
                avi.open(param.toLocal8Bit().begin(),I.width(),I.height(), "MJPG", 30/*fps*/);
            QByteArray data = buffer.data();
            avi.add_frame((unsigned char*)&*data.begin(),data.size(),true);
        }
        avi.close();
        resize(ow,oh);
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
                "Video file (*.avi);;All files (*)");
    if(filename.isEmpty())
        return;
    bool ok;
    int angle = QInputDialog::getInt(this,
            "DSI Studio",
            "Assign angle increament in degree(s):",1,1,10,1,&ok);
    if(!ok)
        return;
    command("save_rotation_video",filename,QString::number(angle));
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

    paintGL();
}

void GLWidget::rotate(void)
{
    int now_time = time.elapsed();
    rotate_angle((now_time-last_time)/100.0,0,1.0,0.0);
    last_time = now_time;
    updateGL();
}



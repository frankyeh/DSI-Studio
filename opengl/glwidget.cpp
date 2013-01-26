#define NOMINMAX
#include <QtOpenGL>
#include <QtGui>
#include <QMessageBox>
#include <QInputDialog>
#include <QTimer>
#include <QClipboard>
#include <math.h>
#include <vector>
#include "math/matrix_op.hpp"
#include "glwidget.h"
#include "tracking_static_link.h"
#include "tracking/tracking_window.h"
#include "ui_tracking_window.h"
#include "renderingtablewidget.h"
#include "tracking/region/regiontablewidget.h"
#include "SliceModel.h"
#include "libs/tracking/tracking_model.hpp"


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
        editing_option(0),
        current_visible_slide(0),
        scaled_factor(1.0),
        set_view_flip(false)
{
    std::fill(slice_texture,slice_texture+3,0);
    odf_dim = 0;
    odf_slide_pos = 0;
    max_fa = *std::max_element(cur_tracking_window.slice.source_images.begin(),cur_tracking_window.slice.source_images.end());
    if(max_fa == 0.0)
        max_fa = 1.0;
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



void GLWidget::initializeGL()
{
    makeCurrent();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
    glBlendFunc (GL_DST_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    tracts = glGenLists(1);
    set_view(2/*axial view*/);
    check_error(__FUNCTION__);
}

void GLWidget::set_view(unsigned char view_option)
{
    // initialize world matrix
    math::matrix_identity(transformation_matrix,math::dim<4,4>());
    if(cur_tracking_window.slice.voxel_size[0] > 0.0)
    {
        transformation_matrix[5] = cur_tracking_window.slice.voxel_size[1] / cur_tracking_window.slice.voxel_size[0];
        transformation_matrix[10] = cur_tracking_window.slice.voxel_size[2] / cur_tracking_window.slice.voxel_size[0];
    }

    transformation_matrix[0] *= scaled_factor;
    transformation_matrix[5] *= scaled_factor;
    transformation_matrix[10] *= scaled_factor;
    transformation_matrix[12] = -transformation_matrix[0]*cur_tracking_window.slice.center_point[0];
    transformation_matrix[13] = -transformation_matrix[5]*cur_tracking_window.slice.center_point[1];
    transformation_matrix[14] = -transformation_matrix[10]*cur_tracking_window.slice.center_point[2];

    if(view_option != 2)
    {
        float m1[16],m2[16];
        std::copy(transformation_matrix,transformation_matrix+16,m1);
        std::fill(m2, m2 + 16, 0.0);
        m2[15] = 1.0;
        switch(view_option)
        {
        case 0:
            m2[2] = -1.0;
            m2[4] = 1.0;
            m2[9] = -1.0;
            break;
        case 1:
            m2[0] = 1.0;
            m2[6] = 1.0;
            m2[9] = -1.0;
            break;
        case 2:
            break;
        }
        math::matrix_product(m1, m2, transformation_matrix, math::dim<4, 4>(),math::dim<4, 4>());
    }
    // rotate 180 degrees
    if(set_view_flip)
    {
        float m1[16],m2[16];
        std::copy(transformation_matrix,transformation_matrix+16,m1);
        std::fill(m2, m2 + 16, 0.0);
        m2[0] = -1.0;
        m2[5] = 1.0;
        m2[10] = -1.0;
        m2[15] = 1.0;
        math::matrix_product(m1, m2, transformation_matrix, math::dim<4, 4>(),math::dim<4, 4>());
    }
    set_view_flip = !set_view_flip;
}

void setupLight(float ambient,float diffuse,float angle,float angle1,unsigned char light_option)
{
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

void setupMaterial(float emission)
{
    GLfloat material2[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
    std::fill(material2,material2+3,emission);
    glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,material2);

    std::fill(material2,material2+3,emission);
    glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,material2);
    check_error(__FUNCTION__);
}

unsigned char getCurView(float* transformation_matrix)
{
    unsigned char cur_view = 0;
    {
        const float view_dirs[6][3] = {{1,0,0},{0,1,0},{0,0,1},{-1,0,0},{0,-1,0},{0,0,-1}};
        float mat[16],view[16];
        //math::matrix_product(transformation_matrix,mat,view,math::dim<4,4>(),math::dim<4,4>());
        math::matrix_inverse(transformation_matrix,mat,math::dim<4,4>());
        image::vector<3,float> dir(mat+8);
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

void GLWidget::paintGL()
{

    //if(!cur_tracking_window.ui)
    //    return;
    // return to the original view matrix
    makeCurrent();
    /*
    QPainter p(this);
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    */

    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        //gluPerspective(get_param("fov_angle")*5+1,
        //               (float)width/(float)height,1.0,1000.0);
        // The following code is a fancy bit of math that is eqivilant to calling:
        // gluPerspective( fieldOfView/2.0f, width/height , 0.1f, 255.0f )
        // We do it this way simply to avoid requiring glu.h
        GLfloat fieldOfView = 2.0*(get_param_float("fov_angle")*5.0+1);
        GLfloat zNear = 1.0f;
        GLfloat zFar = 1000.0f;
        GLfloat aspect = float(width)/float(height);
        GLfloat fH = tan( float(fieldOfView / 360.0f * 3.14159f) ) * zNear;
        GLfloat fW = fH * aspect;
        glFrustum( -fW, fW, -fH, fH, zNear, zFar );

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        my_gluLookAt(0,0,get_param_float("view_distance")*-40,0,0,0,0,-1.0,0);
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    //glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);


    {

        int color = get_param("bkg_color");
        glClearColor((float)((color & 0x00FF0000) >> 16)/255.0,
                     (float)((color & 0x0000FF00) >> 8)/255.0,
                     (float)(color & 0x000000FF)/255.0,1.0f);

        if(get_param("anti_aliasing"))
            glEnable(0x809D/*GL_MULTISAMPLE*/);
        else
            glDisable(0x809D/*GL_MULTISAMPLE*/);

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

    if (cur_tracking_window.has_odfs &&
        get_param("show_odf"))
    {
        ODFModel* handle = cur_tracking_window.handle;
        SliceModel& slice = cur_tracking_window.slice;
        if(!get_param("odf_position"))
        {
        if(odf_dim != slice.cur_dim ||
           odf_slide_pos != slice.slice_pos[slice.cur_dim])
        {
            odf_dim = slice.cur_dim;
            odf_slide_pos = slice.slice_pos[odf_dim];
            odf_points.clear();
            image::geometry<2> geo(slice.geometry[odf_dim==0?1:0],
                                   slice.geometry[odf_dim==2?1:2]);
            float fa[3],dir[9];
            int sX, sY, sZ;
            float fa_threshold = cur_tracking_window.ui->fa_threshold->value();
            unsigned char skip_mask_set[3] = {0,1,3};
            unsigned char mask = skip_mask_set[get_param("odf_skip")];
            for(image::pixel_index<2> index;index.valid(geo);index.next(geo))
            {
                if((index[0] & mask) | (index[1] & mask))
                    continue;
                if (!slice.get3dPosition(index[0],index[1],sX,sY,sZ) ||
                !tracking_get_voxel_dir(cur_tracking_window.handle, sX, sY, sZ, fa,dir) ||
                fa[0] <= fa_threshold)
                    continue;
                add_odf(sX,sY,sZ);
            }
        }
        }
        else
        {
            odf_points.clear();
            add_odf(slice.slice_pos[0],
                    slice.slice_pos[1],
                    slice.slice_pos[2]);
        }
        if(odf_colors.empty())
        {
            for (unsigned int index = 0; index <
                 cur_tracking_window.odf_size; ++index)
            {
                odf_colors.push_back(std::abs(get_odf_direction(handle, index)[0]));
                odf_colors.push_back(std::abs(get_odf_direction(handle, index)[1]));
                odf_colors.push_back(std::abs(get_odf_direction(handle, index)[2]));
            }
        }
        glEnable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);
        glPushMatrix();
        glMultMatrixf(transformation_matrix);
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
                           GL_UNSIGNED_SHORT,get_odf_faces(handle,0));
        }
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
                   (float)(get_param("tract_light_dir"))*3.1415926*2.0/10.0,
                   (float)(get_param("tract_light_shading"))*3.1415926/20.0,
                   get_param("tract_light_option"));

        glPushMatrix();
        glMultMatrixf(transformation_matrix);
        setupMaterial((float)(get_param("tract_emission"))/10.0);

        if(get_param("tract_color_style") != tract_color_style)
        {
            if(get_param("tract_color_style") > 1 &&
                    get_param("tract_color_style") <= 3) // index painting
            {
                cur_tracking_window.ui->color_bar->show();
                cur_tracking_window.ui->color_bar->setFloating(true);
                cur_tracking_window.ui->color_bar->setAllowedAreas( Qt::NoDockWidgetArea );
            }
            else
                cur_tracking_window.ui->color_bar->hide();
        }


        if(get_param_float("tract_alpha") != tract_alpha ||
           get_param("tract_alpha_style") != tract_alpha_style ||
           get_param("tract_style") != tract_style ||
           get_param("tract_color_style") != tract_color_style ||
           get_param("tract_size") != tube_size ||
           get_param("tract_tube_detail") != tract_tube_detail ||
           get_param("end_point_shift") != end_point_shift)
        {
            tract_alpha = get_param_float("tract_alpha");
            tract_alpha_style = get_param("tract_alpha_style");
            tract_style = get_param("tract_style");
            tract_color_style = get_param("tract_color_style");
            tube_size = get_param("tract_size");
            tract_tube_detail = get_param("tract_tube_detail");
            end_point_shift = get_param("end_point_shift");
            makeTracts();
        }
        if(get_param_float("tract_alpha") != 10.0)
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
        glCallList(tracts);
        glPopMatrix();
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);
        glDepthMask(true);
        check_error("show_tract");
    }

    if (get_param("show_slice"))
    {
        if(current_visible_slide)
        {
            unsigned int index = current_visible_slide-1;
            if(!mi3s[index].ended)
            {
                slice_contrast = cur_tracking_window.ui->gl_contrast_value->value();
                slice_offset = cur_tracking_window.ui->gl_offset_value->value();
                slice_index = current_visible_slide;
                const float* buf = mi3s[index].get();
                std::vector<float> inverse_transform(16);
                image::create_affine_transformation_matrix(buf, buf + 9,inverse_transform.begin(), image::vdim<3>());
                math::matrix_inverse(inverse_transform.begin(),transform[index].begin(),math::dim<4, 4>());
                std::fill(other_slices[index].texture_need_update,
                          other_slices[index].texture_need_update+3,1);

                // update roi image
                {
                    image::basic_image<float,3>& cur_roi_image = roi_image[index];
                    for(image::pixel_index<3> pos;pos.valid(cur_roi_image.geometry());pos.next(cur_roi_image.geometry()))
                    {
                        image::vector<3,float> from(pos),to;
                        image::vector_transformation(from.begin(), to.begin(),inverse_transform.begin(),image::vdim<3>());
                        image::linear_estimate(other_slices[index].source_images,to,cur_roi_image[pos.index()]);
                    }
                }
            }
        }



        glEnable(GL_TEXTURE_2D);
        glEnable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);
        float alpha = get_param_float("slice_alpha")/10.0;
        handleAlpha(image::rgb_color(0,0,0,255),
                        alpha,get_param("slice_bend1"),get_param("slice_bend2"));
        glDepthMask((alpha == 1.0));

        glPushMatrix();
        glMultMatrixf(transformation_matrix);

        std::vector<image::vector<3,float> > points(4);
        SliceModel* active_slice = current_visible_slide ?
                                   (SliceModel*)&other_slices[current_visible_slide-1] :
                                   (SliceModel*)&cur_tracking_window.slice;

        if(cur_tracking_window.ui->gl_contrast_value->value() != slice_contrast ||
           cur_tracking_window.ui->gl_offset_value->value() != slice_offset ||
            slice_index != current_visible_slide)
        {
            slice_contrast = cur_tracking_window.ui->gl_contrast_value->value();
            slice_offset = cur_tracking_window.ui->gl_offset_value->value();
            slice_index = current_visible_slide;
            active_slice->texture_need_update[0] = true;
            active_slice->texture_need_update[1] = true;
            active_slice->texture_need_update[2] = true;
        }

        bool show_slice[3];
        show_slice[0] = cur_tracking_window.ui->glSagCheck->checkState();
        show_slice[1] = cur_tracking_window.ui->glCorCheck->checkState();
        show_slice[2] = cur_tracking_window.ui->glAxiCheck->checkState();

        for(unsigned char dim = 0;dim < 3;++dim)
        {
            if(!show_slice[dim])
                continue;

            if(active_slice->texture_need_update[dim])
            {
                if(slice_texture[dim])
                    deleteTexture(slice_texture[dim]);
                image::color_image texture;
                active_slice->get_texture(dim,texture,slice_contrast,slice_offset);
                slice_texture[dim] =
                    bindTexture(QImage((unsigned char*)&*texture.begin(),
                texture.width(),texture.height(),QImage::Format_RGB32));
            }

            glBindTexture(GL_TEXTURE_2D, slice_texture[dim]);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);

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
                   (float)(get_param("region_light_dir"))*3.1415926*2.0/10.0,
                   (float)(get_param("region_light_shading"))*3.1415926/20.0,
                   get_param("region_light_option"));

        glPushMatrix();
        glMultMatrixf(transformation_matrix);

        setupMaterial((float)(get_param("region_emission"))/10.0);

        float alpha = get_param_float("region_alpha")/10.0;
        unsigned char cur_view = (alpha == 1.0 ? 0 : getCurView(transformation_matrix));
        for(unsigned int index = 0;index < cur_tracking_window.regionWidget->regions.size();++index)
            if(cur_tracking_window.regionWidget->item(index,0)->checkState() == Qt::Checked)
            {
                cur_tracking_window.regionWidget->regions[index].makeMeshes(get_param("region_mesh_smoothed"));
                drawRegion(cur_tracking_window.regionWidget->regions[index].show_region,
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
                   (float)(get_param("surface_light_dir"))*3.1415926*2.0/10.0,
                   (float)(get_param("surface_light_shading"))*3.1415926/20.0,
                   get_param("surface_light_option"));

        glPushMatrix();
        glMultMatrixf(transformation_matrix);
        setupMaterial((float)(get_param("surface_emission"))/10.0);

        float alpha = get_param_float("surface_alpha")/10.0;
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

}

void GLWidget::add_odf(int x,int y,int z)
{
    ODFModel* handle = cur_tracking_window.handle;
    image::vector<3,float> pos(x,y,z);

    const float* odf_buffer =
            handle->fib_data.fib.get_odf_data(
                image::pixel_index<3>(pos[0],pos[1],pos[2],cur_tracking_window.slice.geometry).index());


    if(!odf_buffer)
        return;
    static float size_set[] = {0.5,1.0,1.5,2.0,4.0,8.0,16.0,32.0};
    float scaling = size_set[get_param("odf_size")]*2.0/max_fa;
    unsigned int odf_size = cur_tracking_window.odf_size;
    unsigned int half_odf = odf_size >> 1;
    odf_points.resize(odf_points.size()+odf_size);
    std::vector<image::vector<3,float> >::iterator iter = odf_points.end()-odf_size;
    std::vector<image::vector<3,float> >::iterator end = odf_points.end();
    std::fill(iter,end,pos);

    // smooth the odf a bit

    std::vector<float> new_odf_buffer;
    if(get_param("odf_smoothing"))
    {
        new_odf_buffer.resize(half_odf);
        std::copy(odf_buffer,odf_buffer+half_odf,new_odf_buffer.begin());
        std::vector<image::vector<3,unsigned short> >& odf_faces =
                handle->fib_data.fib.odf_faces;
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
            float sum = odf_buffer[f1]+odf_buffer[f2]+odf_buffer[f3];
            sum *= 0.1;
            if(odf_buffer[f1] == 0.0)
                new_odf_buffer[f1] = std::max(sum,new_odf_buffer[f1]);
            if(odf_buffer[f2] == 0.0)
                new_odf_buffer[f2] = std::max(sum,new_odf_buffer[f2]);
            if(odf_buffer[f3] == 0.0)
                new_odf_buffer[f3] = std::max(sum,new_odf_buffer[f3]);
        }
        odf_buffer = &new_odf_buffer[0];
    }


    float odf_min = *std::min_element(odf_buffer,odf_buffer+half_odf);
    for(unsigned int index = 0;index < half_odf;++index,++iter)
    {
        image::vector<3,float> displacement(get_odf_direction(handle, index));
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
    float alpha = (tract_alpha_style == 0)? tract_alpha/20.0:tract_alpha/10.0;
    const float radius_option[] = {0.01,0.02,0.04,0.08,0.1,0.2,0.4,0.6,0.8};
    const float detail_option[] = {1.0,0.5,0.25,0.0,0.0};
    float radius = radius_option[tube_size];
    bool show_end_points = tract_style == 2;
    float cull_angle_cos = std::cos(cur_tracking_window.ui->turning_angle->value() * 3.1415926 / 180.0);
    float tube_detail = radius*detail_option[tract_tube_detail]*4.0;
    float skip_rate = 1.0;

    float color_r;
    std::vector<float> mean_fa;
    int color_item_index;
    unsigned int mean_fa_index = 0;
    float color_max_value = cur_tracking_window.ui->tract_color_max_value->value();
    float color_min_value = cur_tracking_window.ui->tract_color_min_value->value();
    unsigned int tract_color_index = cur_tracking_window.ui->tract_color_index->currentIndex();

    // show tract by index value
    if (tract_color_style > 1 && tract_color_style <= 3)
    {
        if(tract_color_index > 0)
            color_item_index = cur_tracking_window.handle->fib_data.other_mapping_index+tract_color_index-1;
        if(tract_color_style == 3)// mean value
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
                    if(tract_color_index == 0)
                        cur_tracking_window.handle->get_tract_fa(
                            active_tract_model->get_tract(data_index),color_min_value,cull_angle_cos,fa_values);
                    else
                        cur_tracking_window.handle->get_tract_data(
                            active_tract_model->get_tract(data_index),color_item_index,fa_values);

                    float sum = std::accumulate(fa_values.begin(),fa_values.end(),0.0f);
                    sum /= (float)fa_values.size();
                    mean_fa.push_back(sum);
                }
            }
        }

        color_r = color_max_value-color_min_value;
        if(color_r + 1.0 == 1.0)
            color_r = 1.0;
    }


    std::vector<image::vector<3,float> > points(8),previous_points(8),
                                      normals(8),previous_normals(8);
    image::rgb_color paint_color;
    image::vector<3,float> paint_color_f;
    std::vector<float> color;

    unsigned int visible_tracts_list[] = {5000,10000,20000,50000,100000};
    unsigned int visible_tracts = visible_tracts_list[get_param("tract_visible_tracts")];
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
    boost::uniform_real<> dist(0, 1.0);
    boost::mt19937 gen;
    boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
            uniform_gen(gen, dist);

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
                    if(tract_color_index == 0)
                        cur_tracking_window.handle->get_tract_fa(
                            active_tract_model->get_tract(data_index),color_min_value,cull_angle_cos,color);
                    else
                        cur_tracking_window.handle->get_tract_data(
                            active_tract_model->get_tract(data_index),color_item_index,color);
                    break;
                case 3:// mean
                    paint_color_f =
                            cur_tracking_window.color_map[std::floor(
                                    std::min(1.0f,(std::max<float>(mean_fa[mean_fa_index++]-color_min_value,0.0))/color_r)*255.0+0.49)];
                    break;
                case 4:// mean directional
                    {
                        const float* iter = data_iter;
                        for (unsigned int index = 0;
                             index < vertex_count;iter += 3, ++index)
                        {
                            image::vector<3,float> vec_n;
                            if (index + 1 < vertex_count)
                            {
                                vec_n[0] = iter[3] - iter[0];
                                vec_n[1] = iter[4] - iter[1];
                                vec_n[2] = iter[5] - iter[2];
                                vec_n.normalize();
                                paint_color_f[0] += std::fabs(vec_n[0]);
                                paint_color_f[1] += std::fabs(vec_n[1]);
                                paint_color_f[2] += std::fabs(vec_n[2]);
                            }
                        }
                        paint_color_f /= vertex_count;
                    }
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
                        cur_color = paint_color_f;
                        break;
                    case 2://local anisotropy
                        if(index < color.size())
                        cur_color = cur_tracking_window.color_map[
                        std::floor(std::min(1.0f,(std::max<float>(color[index]-color_min_value,0.0))/color_r)*255.0+0.49)];
                        break;
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
                    vec_ab *= radius;
                    vec_ba *= radius;
                    vec_a *= radius;
                    vec_b *= radius;

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
                            shift *= -end_point_shift;
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
                                shift *= end_point_shift;
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
    makeCurrent();
    width = width_;
    height = height_;
    glViewport(0,0, width, height);
    glMatrixMode(GL_MODELVIEW);
    check_error(__FUNCTION__);
}
void GLWidget::wheelEvent ( QWheelEvent * event )
{
    makeCurrent();
    glPushMatrix();
    glLoadIdentity();
    double scalefactor = event->delta();
    scalefactor /= 1200.0;
    scalefactor = 1.0+scalefactor;
    glScaled(scalefactor,scalefactor,scalefactor);
    scaled_factor *= scalefactor;
    glMultMatrixf(transformation_matrix);
    glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix);
    glPopMatrix();
    updateGL();
    event->ignore();
}
void GLWidget::keyPressEvent ( QKeyEvent * event )
{
    switch(event->key())
    {
    case Qt::Key_Q:
        cur_tracking_window.ui->glSagSlider->setValue(cur_tracking_window.ui->glSagSlider->value()+1);
        break;
    case Qt::Key_A:
        cur_tracking_window.ui->glSagSlider->setValue(cur_tracking_window.ui->glSagSlider->value()-1);
        break;
    case Qt::Key_W:
        cur_tracking_window.ui->glCorSlider->setValue(cur_tracking_window.ui->glCorSlider->value()+1);
        break;
    case Qt::Key_S:
        cur_tracking_window.ui->glCorSlider->setValue(cur_tracking_window.ui->glCorSlider->value()-1);
        break;
    case Qt::Key_E:
        cur_tracking_window.ui->glAxiSlider->setValue(cur_tracking_window.ui->glAxiSlider->value()+1);
        break;
    case Qt::Key_D:
        cur_tracking_window.ui->glAxiSlider->setValue(cur_tracking_window.ui->glAxiSlider->value()-1);
        break;
    case Qt::Key_Z:
        set_view(0);
        updateGL();
        break;
    case Qt::Key_X:
        set_view(1);
        updateGL();
        break;
    case Qt::Key_C:
        set_view(2);
        updateGL();
        break;
    }
}

void GLWidget::slice_location(unsigned char dim,std::vector<image::vector<3,float> >& points)
{
    SliceModel* active_slice = current_visible_slide ?
                               (SliceModel*)&other_slices[current_visible_slide-1] :
                               (SliceModel*)&cur_tracking_window.slice;
    active_slice->get_slice_positions(dim,points);
    if(current_visible_slide)
    for(unsigned int index = 0;index < 4;++index)
    {
        image::vector<3,float> tmp;
        image::vector_transformation(points[index].begin(), tmp.begin(),
            transform[current_visible_slide-1].begin(), image::vdim<3>());
        points[index] = tmp;
    }
}

void GLWidget::get_view_dir(QPoint p,image::vector<3,float>& dir)
{
    float m[16],v[3];
    glGetFloatv(GL_PROJECTION_MATRIX,m);
    // Compute the vector of the pick ray in screen space
    v[0] = (( 2.0f * ((float)p.x())/((float)width)) - 1 ) / m[0];
    v[1] = -(( 2.0f * ((float)p.y())/((float)height)) - 1 ) / m[5];
    v[2] = -1.0f;
    // Transform the screen space pick ray into 3D space
    dir[0] = v[0]*mat[0] + v[1]*mat[4] + v[2]*mat[8];
    dir[1] = v[0]*mat[1] + v[1]*mat[5] + v[2]*mat[9];
    dir[2] = v[0]*mat[2] + v[1]*mat[6] + v[2]*mat[10];
    dir.normalize();
}

bool get_slice_projection_point(const std::vector<image::vector<3,float> >& slice_points,
                                const image::vector<3,float>& pos,
                                const image::vector<3,float>& dir,
                                float& dx,float& dy,float& dis)
{
    image::vector<3,float> pos_offset(pos),v1(slice_points[1]),v2(slice_points[2]),v3(dir);
    pos_offset -= slice_points[0];
    v1 -= slice_points[0];
    v2 -= slice_points[0];
    float m[9],result[3];
    m[0] = v1[0];
    m[1] = v2[0];
    m[2] = -v3[0];
    m[3] = v1[1];
    m[4] = v2[1];
    m[5] = -v3[1];
    m[6] = v1[2];
    m[7] = v2[2];
    m[8] = -v3[2];

    if(!math::matrix_inverse(m,math::dim<3,3>()))
        return false;
    math::matrix_vector_product(m,pos_offset.begin(),result,math::dim<3,3>());
    dx = result[0];
    dy = result[1];
    dis = result[2];
    return true;
}
bool get_slice_projection_point(const std::vector<image::vector<3,float> >& slice_points,
                                const image::vector<3,float>& pos,
                                const image::vector<3,float>& dir,
                                float& dx,float& dy)
{
    float dis;
    return get_slice_projection_point(slice_points,pos,dir,dx,dy,dis);
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
    if(!cur_tracking_window.slice.geometry.is_valid(voxel))
        continue;
    for(int index = 0;index < cur_tracking_window.regionWidget->regions.size();++index)
        if(cur_tracking_window.regionWidget->regions[index].has_point(voxel) &&
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
    std::vector<image::vector<3,float> > points(4);

    // select slice
    slice_selected = false;
    {
        // now check whether the slices are selected
        slice_distance = std::numeric_limits<float>::max();
        for(unsigned char dim = 0;dim < 3;++dim)
        {
            if(!show_slice[dim])
                continue;
            slice_location(dim,points);
            float d;
            if(!get_slice_projection_point(points,pos,dir1,slice_dx,slice_dy,d))
                continue;
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
void GLWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    makeCurrent();
    {
        float view[16];
        //glMultMatrixf(transformation_matrix);
        glGetFloatv(GL_MODELVIEW_MATRIX,mat);
        math::matrix_product(transformation_matrix,mat,view,math::dim<4,4>(),math::dim<4,4>());
        math::matrix_inverse(view,mat,math::dim<4,4>());
        pos[0] = mat[12];
        pos[1] = mat[13];
        pos[2] = mat[14];
    }
    get_view_dir(event->pos(),dir1);
    // nothing selected
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

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    setFocus();// for key stroke to work
    makeCurrent();
    lastPos = event->pos();
    if(editing_option)
    {
        float view[16];
        //glMultMatrixf(transformation_matrix);
        glGetFloatv(GL_MODELVIEW_MATRIX,mat);
        math::matrix_product(transformation_matrix,mat,view,math::dim<4,4>(),math::dim<4,4>());
        math::matrix_inverse(view,mat,math::dim<4,4>());
        pos[0] = mat[12];
        pos[1] = mat[13];
        pos[2] = mat[14];
    }

    if(editing_option == 2) // move object
    {
        get_view_dir(lastPos,dir1);
        dir1.normalize();

        // nothing selected
        select_object();
        select_slice();

        if(!object_selected && !slice_selected)
        {
            editing_option = 0;
            setCursor(Qt::ArrowCursor);
            return;
        }
        // if only slice is selected or slice is at the front, then move slice
        if(!object_selected || object_distance > slice_distance)
        {
            editing_option = 3;
            return;
        }
        cur_tracking_window.regionWidget->selectRow(selected_index);
        // determine the moving direction of the region
        float angle[3] = {0,0,0};
        bool show_slice[3];
        show_slice[0] = cur_tracking_window.ui->glSagCheck->checkState();
        show_slice[1] = cur_tracking_window.ui->glCorCheck->checkState();
        show_slice[2] = cur_tracking_window.ui->glAxiCheck->checkState();
        std::vector<image::vector<3,float> > points(4);

        for(unsigned char dim = 0;dim < 3;++dim)
        {
            slice_location(dim,points);
            angle[dim] = std::fabs(dir1*get_norm(points)) + (show_slice[dim] ? 1:0);
        }
        moving_at_slice_index = std::max_element(angle,angle+3)-angle;
        slice_location(moving_at_slice_index,points);
        if(!get_slice_projection_point(points,pos,dir1,slice_dx,slice_dy))
        {
            editing_option = 0;
            setCursor(Qt::ArrowCursor);
            return;
        }
        accumulated_dis = math::zero<float>();
    }
}
void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    //set as copytarget
    cur_tracking_window.copy_target =0;
    makeCurrent();
    if(!editing_option)
        return;
    setCursor(Qt::ArrowCursor);
    if(editing_option >= 2)
    {
        editing_option = 0;
        return;
    }
    get_view_dir(lastPos,dir1);
    get_view_dir(QPoint(event->x(),event->y()),dir2);
    editing_option = 0;
    emit edited();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    makeCurrent();
    // move object
    if(editing_option == 2)
    {

        std::vector<image::vector<3,float> > points(4);
        slice_location(moving_at_slice_index,points);
        get_view_dir(event->pos(),dir2);
        float dx,dy;
        if(!get_slice_projection_point(points,pos,dir2,dx,dy))
            return;
        image::vector<3,float> v1(points[1]),v2(points[2]),dis;
        v1 -= points[0];
        v2 -= points[0];
        dis = v1*(dx-slice_dx)+v2*(dy-slice_dy);
        dis -= accumulated_dis;
        image::vector<3,short> apply_dis(dis);
        if(apply_dis[0] != 0 || apply_dis[1] != 0 || apply_dis[2] != 0)
        {
            cur_tracking_window.regionWidget->regions[cur_tracking_window.regionWidget->currentRow()].shift(apply_dis);
            accumulated_dis += apply_dis;
            emit region_edited();
        }
        return;
    }
    // move slice
    if(editing_option == 3)
    {

        std::vector<image::vector<3,float> > points(4);
        slice_location(moving_at_slice_index,points);
        get_view_dir(event->pos(),dir2);
        float move_dis = (dir2-dir1)*get_norm(points);
        move_dis *= slice_distance;
        if(std::fabs(move_dis) < 1.0)
            return;
        move_dis = std::floor(move_dis+0.5);
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
    // go tract editing
    if(editing_option)
        return;

    float dx = event->x() - lastPos.x();
    float dy = event->y() - lastPos.y();
    if(event->modifiers() & Qt::ControlModifier)
    {
        dx /= 16.0;
        dy /= 16.0;
    }

    glPushMatrix();
    glLoadIdentity();
    // the joystick mode is the second step transformation
    if (event->buttons() & Qt::LeftButton)
    {
        glRotated(dy / 2.0, 1.0, 0.0, 0.0);
        if(!(event->modifiers() & Qt::ShiftModifier))
            glRotated(-dx / 2.0, 0.0, 1.0, 0.0);
    }
    else if (event->buttons() & Qt::RightButton)
    {
        double scalefactor = (-dx+dy+100.0)/100.0;
        glScaled(scalefactor,scalefactor,scalefactor);
        scaled_factor *= scalefactor;
    }
    else
        glTranslated(dx/5.0,dy/5.0,0);

    glMultMatrixf(transformation_matrix);
    glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix);
    glPopMatrix();
    updateGL();
    lastPos = event->pos();
}

void GLWidget::saveCamera(void)
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save Translocation Matrix",
            cur_tracking_window.absolute_path + "/camera.txt",
            "Text files (*.txt);;All files (*.*)");
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
            "Open Translocation Matrix",
            cur_tracking_window.absolute_path,
            "Text files (*.txt);;All files (*.*)");
    std::ifstream in(filename.toLocal8Bit().begin());
    if(filename.isEmpty() || !in)
        return;
    std::vector<float> data;
    std::copy(std::istream_iterator<float>(in),
              std::istream_iterator<float>(),std::back_inserter(data));
    data.resize(16);
    std::copy(data.begin(),data.end(),transformation_matrix);
    updateGL();
}
void GLWidget::get_current_slice_transformation(
            image::geometry<3>& geo,image::vector<3,float>& vs,std::vector<float>& tr)
{
    if(!current_visible_slide)
    {
        geo = cur_tracking_window.slice.geometry;
        vs = cur_tracking_window.slice.voxel_size;
        std::fill(tr.begin(),tr.end(),0.0);
        tr[0] = tr[5] = tr[10] = tr[15] = 1.0;
        return;
    }
    geo = other_slices[current_visible_slide-1].geometry;
    vs = other_slices[current_visible_slide-1].voxel_size;
    tr = transform[current_visible_slide-1];
    tr.resize(16);
    tr[15] = 1.0;
    math::matrix_inverse(tr.begin(),math::dim<4,4>());
}

void GLWidget::saveMapping(void)
{
    if(!current_visible_slide)
        return;
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save Mapping Matrix",
            cur_tracking_window.absolute_path + "/mapping.txt",
            "Text files (*.txt);;All files (*.*)");
    if(filename.isEmpty())
        return;
    std::ofstream out(filename.toLocal8Bit().begin());

    for(int row = 0,index = 0;row < 4;++row)
    {
        for(int col = 0;col < 4;++col,++index)
            out << transform[current_visible_slide-1][index] << " ";
        out << std::endl;
    }
}

void GLWidget::loadMapping(void)
{
    if(!current_visible_slide)
        return;
    QString filename = QFileDialog::getOpenFileName(
            this,
            "Open Mapping Matrix",
            cur_tracking_window.absolute_path,
            "Text files (*.txt);;All files (*.*)");
    std::ifstream in(filename.toLocal8Bit().begin());
    if(filename.isEmpty() || !in)
        return;
    mi3s[current_visible_slide-1].terminate();
    std::vector<float> data;
    std::copy(std::istream_iterator<float>(in),
              std::istream_iterator<float>(),std::back_inserter(data));
    data.resize(16);
    std::copy(data.begin(),data.end(),transform[current_visible_slide-1].begin());
    std::fill(other_slices[current_visible_slide-1].texture_need_update,
              other_slices[current_visible_slide-1].texture_need_update+3,1);
    updateGL();
}

bool GLWidget::addSlices(QStringList filenames)
{
    std::vector<std::string> files(filenames.size());
    for (unsigned int index = 0; index < filenames.size(); ++index)
            files[index] = filenames[index].toLocal8Bit().begin();

    image::io::volume volume;
    if(!volume.load_from_files(files,files.size()))
    {
        gz_nifti nifti;
        if(files.size() != 1 || !nifti.load_from_file(files[0]))
        {
            QMessageBox::information(&cur_tracking_window,"DSI Studio","Cannot parse the images",0);
            return false;
        }
        else
            other_slices.push_back(new CustomSliceModel(nifti,cur_tracking_window.slice.center_point));
    }
    else
        other_slices.push_back(new CustomSliceModel(volume,cur_tracking_window.slice.center_point));

    mi3s.push_back(new LinearMapping<image::basic_image<float,3,image::const_pointer_memory<float> >,image::rigid_scaling_transform<3> >);
    mi3s.back().from = cur_tracking_window.slice.source_images;
    mi3s.back().to = other_slices.back().source_images;
    mi3s.back().arg_min.scaling[0] = cur_tracking_window.slice.voxel_size[0] / other_slices.back().voxel_size[0];
    mi3s.back().arg_min.scaling[1] = cur_tracking_window.slice.voxel_size[1] / other_slices.back().voxel_size[1];
    mi3s.back().arg_min.scaling[2] = cur_tracking_window.slice.voxel_size[2] / other_slices.back().voxel_size[2];
    mi3s.back().thread_argmin(image::reg::rigid_body);
    // handle views
    current_visible_slide = mi3s.size();
    transform.push_back(std::vector<float>(16));
    roi_image.push_back(new image::basic_image<float,3>(cur_tracking_window.handle->fib_data.dim));
    roi_image_buf.push_back(&*roi_image.back().begin());
    return true;
}

void GLWidget::delete_slice(int index)
{
    other_slices.erase(other_slices.begin()+index);
    mi3s.erase(mi3s.begin()+index);
    transform.erase(transform.begin()+index);
    roi_image.erase(roi_image.begin()+index);
    roi_image_buf.erase(roi_image_buf.begin()+index);
}

void GLWidget::addSurface(void)
{
    SliceModel* active_slice = current_visible_slide ?
                               (SliceModel*)&other_slices[current_visible_slide-1] :
                               (SliceModel*)&cur_tracking_window.slice;

    float threshold = image::segmentation::otsu_threshold(active_slice->get_source());
    bool ok;
    threshold = QInputDialog::getDouble(this,
        "DSI Studio","Threshold:", threshold,
        *std::min_element(active_slice->get_source().begin(),active_slice->get_source().end()),
        *std::max_element(active_slice->get_source().begin(),active_slice->get_source().end()),
        4, &ok);
    if (!ok)
        return;
    {
        surface.reset(new RegionModel);
        image::basic_image<float, 3> crop_image(active_slice->get_source());
        switch(cur_tracking_window.ui->surfaceStyle->currentIndex())
        {
        case 1: //right hemi
            for(unsigned int index = 0;index < crop_image.size();index += crop_image.width())
            {
                std::fill(crop_image.begin()+index+active_slice->slice_pos[0],
                          crop_image.begin()+index+crop_image.width(),crop_image[0]);
            }
            break;
        case 2: // left hemi
            for(unsigned int index = 0;index < crop_image.size();index += crop_image.width())
            {
                std::fill(crop_image.begin()+index,
                          crop_image.begin()+index+active_slice->slice_pos[0],crop_image[0]);
            }
            break;
        case 3: //lower
            std::fill(crop_image.begin()+active_slice->slice_pos[2]*crop_image.plane_size(),
                      crop_image.end(),crop_image[0]);
            break;
        case 4: // higher
            std::fill(crop_image.begin(),
                      crop_image.begin()+active_slice->slice_pos[2]*crop_image.plane_size(),crop_image[0]);
            break;
        case 5: //anterior
            for(unsigned int index = 0;index < crop_image.size();index += crop_image.plane_size())
            {
                std::fill(crop_image.begin()+index+active_slice->slice_pos[1]*crop_image.width(),
                          crop_image.begin()+index+crop_image.plane_size(),crop_image[0]);
            }
            break;
        case 6: // posterior
            for(unsigned int index = 0;index < crop_image.size();index += crop_image.plane_size())
            {
                std::fill(crop_image.begin()+index,
                          crop_image.begin()+index+active_slice->slice_pos[1]*crop_image.width(),crop_image[0]);
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
            return;
        }
    }

    if(current_visible_slide)
    for(unsigned int index = 0;index < surface->get()->point_list.size();++index)
    {
        image::vector<3,float> tmp;
        image::vector_transformation(
            surface->get()->point_list[index].begin(), tmp.begin(),
            transform[current_visible_slide-1].begin(), image::vdim<3>());
        tmp += 0.5;
        surface->get()->point_list[index] = tmp;
    }
}

void GLWidget::copyToClipboard(void)
{
    updateGL();
    QApplication::clipboard()->setImage(grabFrameBuffer());
}

void GLWidget::catchScreen(void)
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Save Images files",
            cur_tracking_window.absolute_path,
            "PNG files (*.png );;BMP files (*.bmp);;JPEG File (*.jpg);;TIFF File (*.tif);;All files (*.*)");
    if(filename.isEmpty())
        return;
    updateGL();
    grabFrameBuffer().save(filename);
}
void GLWidget::saveLeftRight3DImage(void)
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Assign image name",
            cur_tracking_window.absolute_path,
            "BMP files (*.bmp);;PNG files (*.png );;JPEG File (*.jpg);;TIFF File (*.tif);;All files (*.*)");
    if(filename.isEmpty())
        return;
    bool ok;
    int angle = QInputDialog::getInteger(this,
            "DSI Studio",
            "Assign left angle difference in degrees (negative value for right/left view)):",5,-60,60,5,&ok);
    if(!ok)
        return;
    makeCurrent();
    glPushMatrix();

    glLoadIdentity();
    glRotated(angle/2.0, 0.0, 1.0, 0.0);
    glMultMatrixf(transformation_matrix);
    glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix);
    updateGL();
    QImage left = grabFrameBuffer();
    glLoadIdentity();
    glRotated(-angle, 0.0, 1.0, 0.0);
    glMultMatrixf(transformation_matrix);
    glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix);
    updateGL();
    QImage right = grabFrameBuffer();
    glPopMatrix();
    QImage all(left.width()*2,left.height(),QImage::Format_ARGB32);
    QPainter painter(&all);
    painter.drawImage(0,0,left);
    painter.drawImage(left.width(),0,right);
    all.save(filename);
}
void GLWidget::saveRotationSeries(void)
{
    QString filename = QFileDialog::getSaveFileName(
            this,
            "Assign image name",
            cur_tracking_window.absolute_path,
            "BMP files (*.bmp);;PNG files (*.png );;JPEG File (*.jpg);;TIFF File (*.tif);;All files (*.*)");
    if(filename.isEmpty())
        return;
    bool ok;
    int angle = QInputDialog::getInteger(this,
        "DSI Studio","Rotation angle in each step (degrees):",10,1,360,5,&ok);
    if(!ok)
        return;
    makeCurrent();
    std::vector<float> m(transformation_matrix,transformation_matrix+16);
    begin_prog("save images");
    ::can_cancel(true);
    for(unsigned int index = 0;check_prog(index,360);index += angle)
    {
        glPushMatrix();
        glLoadIdentity();
        glMultMatrixf(transformation_matrix);
        glTranslatef(cur_tracking_window.slice.center_point[0],
                     cur_tracking_window.slice.center_point[1],
                     cur_tracking_window.slice.center_point[2]);
        glRotated(angle, 0.0, 0.0, 1.0);
        glTranslatef(-cur_tracking_window.slice.center_point[0],
                     -cur_tracking_window.slice.center_point[1],
                     -cur_tracking_window.slice.center_point[2]);
        glGetFloatv(GL_MODELVIEW_MATRIX,transformation_matrix);
        glPopMatrix();
        updateGL();
        grabFrameBuffer().save(
                QFileInfo(filename).absolutePath()+"/"+
                QFileInfo(filename).baseName()+"_"+
                QString("%1").arg((int)index,4,10,0,'0')+"."+
                QFileInfo(filename).suffix());
    }
    std::copy(m.begin(),m.end(),transformation_matrix);
    updateGL();
}


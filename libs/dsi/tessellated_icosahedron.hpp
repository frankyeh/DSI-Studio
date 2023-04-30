#ifndef TESSELLATED_ICOSAHEDRON_HPP
#define TESSELLATED_ICOSAHEDRON_HPP
#include "zlib.h"
#include "TIPL/tipl.hpp"
#include <cmath>
#include <vector>
#include <functional>
#ifndef M_PI
#define M_PI        3.14159265358979323846
#endif

extern float odf6_vec[362][3];
extern unsigned short odf6_face[720][3];
extern float odf8_vec[642][3];
extern unsigned short odf8_face[1280][3];

class tessellated_icosahedron
{
public:
    unsigned short fold;
    unsigned short vertices_count;
    unsigned short half_vertices_count;
    std::vector<tipl::vector<3,float> > vertices;
    std::vector<tipl::vector<3,unsigned short> > faces;
    std::vector<std::vector<float> > icosa_cos;
private:
    double face_dis,angle_res;
    unsigned short cur_vertex = 0;
    void add_vertex(const tipl::vector<3,float>& vertex)
    {
        vertices[cur_vertex] = vertex;
        vertices[cur_vertex + half_vertices_count] = -vertex;
		++cur_vertex;
    }
        void get_mid_vertex(const tipl::vector<3,float>& v1,
                                            const tipl::vector<3,float>& v2,
                                                tipl::vector<3,float>& v3) const
	{
		v3 = v1;
		v3 += v2;
		v3.normalize();
	}
        void get_mid_vertex(const tipl::vector<3,float>& v1,
                                            const tipl::vector<3,float>& v2,
                                                const tipl::vector<3,float>& v3,
                                                tipl::vector<3,float>& v4) const
	{
		v4 = v1;
		v4 += v2;
		v4 += v3;
		v4.normalize();
	}
public:
    void get_edge_segmentation2(unsigned short from_vertex,unsigned short to_vertex,
                                   std::vector<unsigned short>& edge)
    {
        edge.push_back(from_vertex);
        tipl::vector<3,float> uv = vertices[to_vertex];
        uv -= vertices[from_vertex];
        double angle = std::acos(1.0-double(uv.length2())*0.5);
        uv.normalize();
        double face_d = std::cos(angle*0.5);
        for (unsigned int index = 1;index < fold;++index)
        {
            double phi = angle*index;
            phi /= double(fold);
            double y = face_d/std::cos(phi-angle*0.5);
            tipl::vector<3,float> vec = uv;
            vec *= std::sqrt(1.0 + y*(y-2.0*std::cos(phi)));
            vec += vertices[from_vertex];
            vec.normalize();
            edge.push_back(cur_vertex);
            add_vertex(vec);
        }
        edge.push_back(to_vertex);
    }

    void get_edge_segmentation(unsigned short from_vertex,unsigned short to_vertex,
                               std::vector<unsigned short>& edge)
    {
		edge.push_back(from_vertex);
        if (fold == 2)
        {
                        tipl::vector<3,float> uv;
			get_mid_vertex(vertices[to_vertex],vertices[from_vertex],uv);
			edge.push_back(cur_vertex);
            add_vertex(uv);
        }
		else
        if(fold == 4)
		{
                        tipl::vector<3,float> mid;
			get_mid_vertex(vertices[to_vertex],vertices[from_vertex],mid);
                        tipl::vector<3,float> u = mid;
			get_mid_vertex(mid,vertices[from_vertex],u);
			edge.push_back(cur_vertex);
            add_vertex(u);
			edge.push_back(cur_vertex);
            add_vertex(mid);
			get_mid_vertex(mid,vertices[to_vertex],u);
			edge.push_back(cur_vertex);
            add_vertex(u);
		}
        else
        {
            tipl::vector<3,float> uv = vertices[to_vertex];
            uv -= vertices[from_vertex];
            uv.normalize();
            for (unsigned int index = 1;index < fold;++index)
            {
                double phi = angle_res*index;
                phi /= double(fold);
                double y = face_dis/std::cos(phi-double(angle_res)*0.5);
                tipl::vector<3,float> vec = uv;
                vec *= std::sqrt(1.0 + y*(y-2.0*std::cos(phi)));
                vec += vertices[from_vertex];
                vec.normalize();

                edge.push_back(cur_vertex);
                add_vertex(vec);
            }
        }
		edge.push_back(to_vertex);
    }
    unsigned short opposite(unsigned short v1) const
	{
        return (v1 < half_vertices_count) ? v1 + half_vertices_count:v1 - half_vertices_count;
	}
    void add_face(unsigned short v1,unsigned short v2,unsigned short v3)
    {
        faces.push_back(tipl::vector<3,unsigned short>(v1,v2,v3));
        faces.push_back(tipl::vector<3,unsigned short>(opposite(v1),opposite(v3),opposite(v2)));
    }
    template<class input_type1,class input_type2>
    void add_faces_in_line(input_type1 up,input_type2 down,unsigned int up_count)
	{
        for(unsigned int index = 0;index < up_count;++index)
        {
            add_face(up[index],down[index],down[index+1]);
			if(index + 1 < up_count)
                add_face(up[index+1],up[index],down[index+1]);
		}
	}
    template<class input_type1,class input_type2,class input_type3>
	void add_faces_in_triangle(input_type1 edge0,
							   input_type2 edge1,
                                                           input_type3 edge2,unsigned int folding)
    {
		if(folding <= 1)
			return;
		if(folding == 2)
		{
			add_face(edge2[1],edge0[0],edge0[1]);
			add_face(edge0[1],edge1[0],edge1[1]);
			add_face(edge1[1],edge2[0],edge2[1]);
			add_face(edge0[1],edge1[1],edge2[1]);
			return;
		}
		std::vector<unsigned short> mid_line(folding);
		mid_line.front() = edge2[folding-1];
		mid_line.back() = edge1[1];
        auto old_cur_vertex = cur_vertex;
        for(unsigned int index = 1;index < mid_line.size()-1;++index,++cur_vertex)
		    mid_line[index] = cur_vertex;
		add_faces_in_triangle(mid_line,edge1+1,edge2,folding-1);
		add_faces_in_line(mid_line,edge0,folding);
        cur_vertex = old_cur_vertex;
	}
	// This function obtain the tessellated points and faces from a icosahedron triangle
    template<class input_type1,class input_type2,class input_type3>
        void build_faces(input_type1 edge0,input_type2 edge1,input_type3 edge2)
    {
        if(fold > 8)
        {
            unsigned short old_fold = fold;
            fold >>= 1;
            std::vector<unsigned short> e0,e1,e2;
            get_edge_segmentation2(edge2[fold],edge1[fold],e0);
            get_edge_segmentation2(edge0[fold],edge2[fold],e1);
            get_edge_segmentation2(edge1[fold],edge0[fold],e2);
            build_faces(e2.begin(),e1.begin(),e0.begin());
            build_faces(edge0,e1.begin(),edge2+fold);
            build_faces(edge1,e2.begin(),edge0+fold);
            build_faces(edge2,e0.begin(),edge1+fold);
            fold = old_fold;
            return;
        }
        add_faces_in_triangle(edge0,edge1,edge2,fold);
        if(fold == 3)
		{
            tipl::vector<3,float> mid;
			get_mid_vertex(vertices[edge0[0]],vertices[edge1[0]],vertices[edge2[0]],mid);
			add_vertex(mid);
			return;
		}
        if(fold == 4)
		{
            tipl::vector<3,float> u;
			get_mid_vertex(vertices[edge0[2]],vertices[edge2[2]],u);
			add_vertex(u);
			get_mid_vertex(vertices[edge0[2]],vertices[edge1[2]],u);
			add_vertex(u);
			get_mid_vertex(vertices[edge1[2]],vertices[edge2[2]],u);
			add_vertex(u);
			return;
		}
        if(fold == 5)
		{
                        tipl::vector<3,float> u1,u2,u3,u4,u5,u6;
			get_mid_vertex(vertices[edge0[0]],vertices[edge0[3]],vertices[edge2[2]],u1);
			get_mid_vertex(vertices[edge1[0]],vertices[edge1[3]],vertices[edge0[2]],u3);
			get_mid_vertex(vertices[edge2[0]],vertices[edge2[3]],vertices[edge1[2]],u6);
			get_mid_vertex(u1,u3,u2);
			get_mid_vertex(u3,u6,u5);
			get_mid_vertex(u6,u1,u4);
			add_vertex(u1);
			add_vertex(u2);
			add_vertex(u3);
			add_vertex(u4);
			add_vertex(u5);
			add_vertex(u6);
			return;
		}
        if(fold == 6)
		{
                        tipl::vector<3,float> u1,u2,u3,u4,u5,u6,u7,u8,u9,u10;// (6-1)*(6-2)/2
			get_mid_vertex(vertices[edge0[0]],vertices[edge1[0]],vertices[edge2[0]],u6);

			get_mid_vertex(vertices[edge0[0]],vertices[edge0[3]],vertices[edge2[3]],u1);
			get_mid_vertex(vertices[edge1[0]],vertices[edge1[3]],vertices[edge0[3]],u4);
			get_mid_vertex(vertices[edge2[0]],vertices[edge2[3]],vertices[edge1[3]],u10);

			get_mid_vertex(u6,vertices[edge0[2]],u2);
			get_mid_vertex(u6,vertices[edge0[4]],u3);

			get_mid_vertex(u6,vertices[edge2[4]],u5);
			get_mid_vertex(u6,vertices[edge1[2]],u7);

			get_mid_vertex(u6,vertices[edge2[2]],u8);
			get_mid_vertex(u6,vertices[edge1[4]],u9);

			add_vertex(u1);
			add_vertex(u2);
			add_vertex(u3);
			add_vertex(u4);
			add_vertex(u5);
			add_vertex(u6);
			add_vertex(u7);
			add_vertex(u8);
			add_vertex(u9);
			add_vertex(u10);
			return;
		}
        if(fold == 8)
		{
            tipl::vector<3,float> u[22]; // (8-1)*(8-2)/2=21, add one for index started from1
			get_mid_vertex(vertices[edge0[4]],vertices[edge2[4]],u[8]);
			get_mid_vertex(vertices[edge1[4]],vertices[edge2[4]],u[17]);
			get_mid_vertex(vertices[edge0[4]],vertices[edge1[4]],u[10]);

			get_mid_vertex(vertices[edge0[2]],vertices[edge2[6]],u[1]);
			get_mid_vertex(vertices[edge0[2]],u[8],u[2]);
			get_mid_vertex(vertices[edge0[4]],u[8],u[3]);
			get_mid_vertex(vertices[edge0[4]],u[10],u[4]);
			get_mid_vertex(vertices[edge0[6]],u[10],u[5]);
			get_mid_vertex(vertices[edge0[6]],vertices[edge1[2]],u[6]);

			get_mid_vertex(vertices[edge2[6]],u[8],u[7]);
			get_mid_vertex(u[10],u[8],u[9]);
			get_mid_vertex(vertices[edge1[2]],u[10],u[11]);

			get_mid_vertex(vertices[edge2[4]],u[8],u[12]);
			get_mid_vertex(u[17],u[8],u[13]);
			get_mid_vertex(u[17],u[10],u[14]);
			get_mid_vertex(vertices[edge1[4]],u[10],u[15]);

			get_mid_vertex(vertices[edge2[4]],u[17],u[16]);
			get_mid_vertex(vertices[edge1[4]],u[17],u[18]);

			get_mid_vertex(vertices[edge2[2]],u[17],u[19]);
			get_mid_vertex(vertices[edge1[6]],u[17],u[20]);
			get_mid_vertex(vertices[edge2[2]],vertices[edge1[6]],u[21]);
            for(unsigned int index = 1;index <= 21;++index)
				add_vertex(u[index]);
			return;
		}

    }

    void build_icosahedron()
    {
        // the top vertex
        add_vertex(tipl::vector<3,float>(0.0f,0.0f,1.0f));
        //central vertices around the upper staggered circles
        float sqrt5 = std::sqrt(5.0f);
        float height = 1.0f/sqrt5;
        float radius = 2.0f/sqrt5;
        for (unsigned int index = 0;index < 5;++index)
            add_vertex(tipl::vector<3,float>(
                                   std::cos(float(index)*float(M_PI)*0.4f)*radius,
                                   std::sin(float(index)*float(M_PI)*0.4f)*radius,height));

        //edge_length = std::sqrt(2.0-2.0/sqrt5);
        face_dis = std::sqrt(0.5+0.5/double(sqrt5));
        angle_res = std::acos(1.0/double(sqrt5));

        std::vector<std::vector<unsigned short> > edges(15);
        // top hat
        get_edge_segmentation(0,1,edges[0]);
        get_edge_segmentation(0,2,edges[1]);
        get_edge_segmentation(0,3,edges[2]);
        get_edge_segmentation(0,4,edges[3]);
        get_edge_segmentation(0,5,edges[4]);
        // the edge of hat
        get_edge_segmentation(1,2,edges[5]);
        get_edge_segmentation(2,3,edges[6]);
        get_edge_segmentation(3,4,edges[7]);
        get_edge_segmentation(4,5,edges[8]);
        get_edge_segmentation(5,1,edges[9]);
        // skirt
        get_edge_segmentation(1,opposite(4),edges[10]);
        get_edge_segmentation(2,opposite(5),edges[11]);
        get_edge_segmentation(3,opposite(1),edges[12]);
        get_edge_segmentation(4,opposite(2),edges[13]);
        get_edge_segmentation(5,opposite(3),edges[14]);

		std::vector<std::vector<unsigned short> > redges(15);
        for(unsigned int index = 0;index < redges.size();++index)
        {
            redges[index] = edges[index];
            tipl::add_constant(redges[index],half_vertices_count);
            tipl::mod_constant(redges[index],vertices_count);

        }
		// hat faces
        build_faces(edges[0].begin(),edges[5].begin(),edges[1].rbegin());
        build_faces(edges[1].begin(),edges[6].begin(),edges[2].rbegin());
        build_faces(edges[2].begin(),edges[7].begin(),edges[3].rbegin());
        build_faces(edges[3].begin(),edges[8].begin(),edges[4].rbegin());
        build_faces(edges[4].begin(),edges[9].begin(),edges[0].rbegin());
		// skirt faces
        build_faces(edges[10].begin(),redges[13].begin(),edges[5].rbegin());
        build_faces(edges[11].begin(),redges[14].begin(),edges[6].rbegin());
        build_faces(edges[12].begin(),redges[10].begin(),edges[7].rbegin());
        build_faces(edges[13].begin(),redges[11].begin(),edges[8].rbegin());
        build_faces(edges[14].begin(),redges[12].begin(),edges[9].rbegin());

    }
	void check_vertex(void)
	{

		std::vector<float> min_cos(vertices.size());
        for(unsigned short i = 0;i < vertices_count;++i)
		{
			float value = 0.0;
            for(unsigned short j = 0;j < vertices_count;++j)
            if(j != i && j != opposite(i) && std::abs(vertices[i]*vertices[j]) > value)
				value = std::abs(vertices[i]*vertices[j]);
			min_cos[i] = value;
		}

	}
	void check_face(void)
	{
        std::vector<unsigned int> count(vertices_count);
		std::vector<float> dis;
                for(unsigned int index = 0;index < faces.size();++index)
		{
            ++count[faces[index][0]];
            ++count[faces[index][1]];
            ++count[faces[index][2]];
			dis.push_back(vertices[faces[index][0]]*vertices[faces[index][1]]);
			dis.push_back(vertices[faces[index][1]]*vertices[faces[index][2]]);
			dis.push_back(vertices[faces[index][2]]*vertices[faces[index][0]]);
		}
	}
	void sort_vertices(void)
	{
        std::vector<tipl::vector<3,float> > sorted_vertices(vertices.begin(),vertices.end());
        std::sort(sorted_vertices.begin(),sorted_vertices.end(),std::greater<tipl::vector<3,float> >());
        for(unsigned int index = 0;index < half_vertices_count;++index)
            sorted_vertices[index+half_vertices_count] = -sorted_vertices[index];
        std::vector<unsigned short> index_map(vertices_count);
        for(unsigned short i = 0;i < vertices_count;++i)
		{
            for(unsigned short j = 0;j < vertices_count;++j)
            if(vertices[i] == sorted_vertices[j])
				{
                    index_map[i] = j;
					break;
				}
		}
        for(unsigned int index = 0;index < faces.size();++index)
		{
            faces[index][0] = index_map[faces[index][0]];
            faces[index][1] = index_map[faces[index][1]];
            faces[index][2] = index_map[faces[index][2]];
		}
		sorted_vertices.swap(vertices);
		std::sort(faces.begin(),faces.end());
	}
public:
    void init(unsigned short vertices_count_,const float* odf_buffer,
              unsigned short faces_count_,const short* face_buffer)
    {
        fold = uint16_t(std::floor(std::sqrt((vertices_count_-2)/10.0)+0.5));
        vertices_count = vertices_count_;
        half_vertices_count = vertices_count_ >> 1;
        cur_vertex = 0;
        vertices.resize(vertices_count);
        faces.resize(faces_count_);
        icosa_cos.clear();
        for (unsigned int index = 0;index < vertices_count;++index,odf_buffer += 3)
            vertices[index] = odf_buffer;
        for (unsigned int  index = 0;index < faces_count_;++index,face_buffer += 3)
            faces[index] = face_buffer;
    }

    tessellated_icosahedron(unsigned short fold_ = 8):fold(fold_)
    {
        vertices_count = fold*fold*10+2;
        half_vertices_count = vertices_count >> 1;
        vertices.resize(vertices_count);
        faces.clear();
        icosa_cos.clear();
        switch(fold)
        {
        case 6:
            for(unsigned int index = 0;index < 362;++index)
            {
                vertices[index][0] = odf6_vec[index][0];
                vertices[index][1] = odf6_vec[index][1];
                vertices[index][2] = odf6_vec[index][2];
            }
            faces.resize(720);
            for(unsigned int index = 0;index < 720;++index)
            {
                faces[index][0] = odf6_face[index][0];
                faces[index][1] = odf6_face[index][1];
                faces[index][2] = odf6_face[index][2];
            }break;
        case 8:
            for(unsigned int index = 0;index < 642;++index)
            {
                vertices[index][0] = odf8_vec[index][0];
                vertices[index][1] = odf8_vec[index][1];
                vertices[index][2] = odf8_vec[index][2];
            }
            faces.resize(1280);
            for(unsigned int index = 0;index < 1280;++index)
            {
                faces[index][0] = odf8_face[index][0];
                faces[index][1] = odf8_face[index][1];
                faces[index][2] = odf8_face[index][2];
            }
            break;
        default:
            build_icosahedron();
            sort_vertices();
        }
        #ifdef _DEBUG
        check_vertex();
        check_face();
        #endif

    }

    float vertices_cos(unsigned int v1,unsigned int v2)
    {
        if(icosa_cos.empty())
        {
            icosa_cos.resize(vertices_count);
            for (unsigned int i = 0; i < vertices_count; ++i)
            {
                icosa_cos[i].resize(vertices_count);
                for (unsigned int j = 0; j < vertices_count; ++j)
                    icosa_cos[i][j] = vertices[i]*vertices[j];
            }
        }
        return icosa_cos[v1][v2];
    }

    /*

    short* ti_faces(unsigned int index)
    {
        return faces[index].begin();
    }

    unsigned int ti_vertices_count(void)
    {
        return vertices_count;
    }


    unsigned int ti_faces_count(void)
    {
        return faces_count;
    }


    float* ti_vertices(unsigned int index)
    {
        return vertices[index].begin();
    }
    */

    void save_to_buffer(std::vector<float>& float_data,std::vector<short>& short_data)
    {
        float_data.resize(vertices_count*3);
        short_data.resize(faces.size()*3);
        for (unsigned int i = 0,index = 0;i < vertices_count;++i,index += 3)
            std::copy(vertices[i].begin(),vertices[i].end(),float_data.begin()+index);
        for (unsigned int i = 0,index = 0;i < faces.size();++i,index += 3)
            std::copy(faces[i].begin(),faces[i].end(),short_data.begin()+index);
    }

    unsigned short discretize(const tipl::vector<3,float>& v)
    {
        unsigned short dir_index = 0;
        float max_value = 0.0;
        for (unsigned int index = 0; index < half_vertices_count; ++index)
        {
            float value = std::abs(vertices[index]*v);
            if (value > max_value)
            {
                max_value = value;
                dir_index = uint16_t(index);
            }
        }
        return dir_index;
    }
};

#endif//TESSELLATED_ICOSAHEDRON_HPP

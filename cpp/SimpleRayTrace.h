#ifndef SIMPLE_RAY_TRACE_HEADER
#define SIMPLE_RAY_TRACE_HEADER

#include "cnpy.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>

struct MyCamModel {
    cv::Mat K;
    cv::Mat RT;
    cv::Mat dist;
};

enum class MyObjectType {
    NOTHING,
    AMBIENT_LIGHT,
    POINT_LIGHT,
    CYLINDER_LIGHT,
    CONE_LIGHT,
    SPHERE,
    PLANE,
    TRIANGLE
};

class MyObject {
public:
    double x = 0;
    double y = 0;
    double z = 0;
    double r = 0;
    double min_inner = 0;
    double x_u = 0;
    double y_u = 0;
    double z_u = 1;
    double xyz_sum = 0;
    double x1 = 0;
    double y1 = 0;
    double z1 = 0;
    double x2 = 0;
    double y2 = 0;
    double z2 = 0;
    double x3 = 0;
    double y3 = 0;
    double z3 = 0;
    double i_diffused = 0;
    double i_specular = 0;
    double ks = 0;
    double kd = 0;
    double ka = 1.0;
    double alpha = 1.0;
    double illumination = 0;
    MyObjectType t = MyObjectType::NOTHING;
    MyObject() {
        t = MyObjectType::NOTHING;
        set_phong_material();
    }
    // MyObject(const MyObject &obj) {} use default copy constructor(shallow copy)
    friend std::ostream& operator<<(std::ostream& os, const MyObject& obj);

    void set_ambient_light(double _illu) {
        t = MyObjectType::AMBIENT_LIGHT;
        illumination = _illu;
    }

    void set_point_light(double _x, 
                         double _y, 
                         double _z, 
                         double _i_diffused, 
                         double _i_specular) {
        t = MyObjectType::POINT_LIGHT;
        x = _x;
        y = _y;
        z = _z;
        i_diffused = _i_diffused;
        i_specular = _i_specular;
    }

    void set_cylinder_light(double _x_start, 
                            double _y_start, 
                            double _z_start, 
                            double _x_u, 
                            double _y_u, 
                            double _z_u, 
                            double _r,
                            double _i_diffused, 
                            double _i_specular) {
        t = MyObjectType::CYLINDER_LIGHT;
        x = _x_start;
        y = _y_start;
        z = _z_start;
        double u_len = _x_u*_x_u+_y_u*_y_u+_z_u*_z_u;
        assert(u_len > 0);
        double inv_u_len = 1./sqrt(u_len);
        x_u = _x_u * inv_u_len;
        y_u = _y_u * inv_u_len;
        z_u = _z_u * inv_u_len;
        r = _r;
        i_diffused = _i_diffused;
        i_specular = _i_specular;
    }

    void set_cone_light(double _x_start, 
                            double _y_start, 
                            double _z_start, 
                            double _x_u, 
                            double _y_u, 
                            double _z_u, 
                            double _angle,
                            double _i_diffused, 
                            double _i_specular) {
        t = MyObjectType::CONE_LIGHT;
        x = _x_start;
        y = _y_start;
        z = _z_start;
        double u_len = _x_u*_x_u+_y_u*_y_u+_z_u*_z_u;
        assert(u_len > 0);
        double inv_u_len = 1./sqrt(u_len);
        x_u = _x_u * inv_u_len;
        y_u = _y_u * inv_u_len;
        z_u = _z_u * inv_u_len;
        assert(_angle > 0);
        if (_angle > 360) {
            min_inner = 0.0;
        } else {
            min_inner = cos(_angle*M_PI/360);
        }
        i_diffused = _i_diffused;
        i_specular = _i_specular;
    }

    void set_sphere(double _x, 
                    double _y, 
                    double _z, 
                    double _r) {
        assert(_r >= 0.);
        t = MyObjectType::SPHERE;
        x = _x;
        y = _y;
        z = _z;
        r = _r;
    }

    void set_plane(double _x_coeff, 
                   double _y_coeff, 
                   double _z_coeff, 
                   double _xyz_sum) {
        t = MyObjectType::PLANE;
        double len_xyz = _x_coeff*_x_coeff+_y_coeff*_y_coeff+_z_coeff*_z_coeff;
        assert(len_xyz > 0);
        double inv_len_xyz = 1./sqrt(len_xyz);
        x = _x_coeff * inv_len_xyz;
        y = _y_coeff * inv_len_xyz;
        z = _z_coeff * inv_len_xyz;
        xyz_sum = _xyz_sum * inv_len_xyz;
    }

    void set_triangle(double _x1, 
                      double _y1, 
                      double _z1, 
                      double _x2, 
                      double _y2, 
                      double _z2, 
                      double _x3, 
                      double _y3, 
                      double _z3) {
        t = MyObjectType::TRIANGLE;
        x1 = _x1;
        y1 = _y1;
        z1 = _z1;
        x2 = _x2;
        y2 = _y2;
        z2 = _z2;
        x3 = _x3;
        y3 = _y3;
        z3 = _z3;
    }

    void set_phong_material(double _ks=0, double _kd=0, double _ka=1, double _alpha=1) {
        ks = _ks; // specular reflection
        kd = _kd; // diffuse reflection
        ka = _ka; // ambient reflection
        alpha = _alpha; // shininess
    }

    std::string to_string() const {
        char m_str[256] = {0};
        char buff[1024] = {0};

        sprintf(m_str, "ks=%f kd=%f ka=%f alpha=%f", ks, kd, ka, alpha);
        if (t == MyObjectType::AMBIENT_LIGHT) {
            sprintf(buff, "{AMBIENT_LIGHT illumination=%f}", illumination);
        } else if (t == MyObjectType::POINT_LIGHT) {
            sprintf(buff, "{POINT_LIGHT x=%f y=%f z=%f i_diffused=%f i_specular=%f}", x, y, z, i_diffused, i_specular);
        } else if (t == MyObjectType::CYLINDER_LIGHT) {
            sprintf(buff, "{CYLINDER_LIGHT x=%f y=%f z=%f x_u=%f y_u=%f z_u=%f r=%f i_diffused=%f i_specular=%f}", x, y, z, x_u, y_u, z_u, r, i_diffused, i_specular);
        } else if (t == MyObjectType::CONE_LIGHT) {
            sprintf(buff, "{CONE_LIGHT x=%f y=%f z=%f x_u=%f y_u=%f z_u=%f min_inner=%f i_diffused=%f i_specular=%f}", x, y, z, x_u, y_u, z_u, min_inner, i_diffused, i_specular);
        } else if (t == MyObjectType::SPHERE) {
            sprintf(buff, "{SPHERE x=%f y=%f z=%f r=%f %s}", x, y, z, r, m_str);
        } else if (t == MyObjectType::PLANE) {
            sprintf(buff, "{PLANE x=%f y=%f z=%f xyz_sum=%f %s}", x, y, z, xyz_sum, m_str);
        } else if (t == MyObjectType::TRIANGLE) {
            sprintf(buff, "{TRIANGLE x1=%f y1=%f z1=%f x2=%f y2=%f z2=%f x3=%f y3=%f z3=%f %s}", x1, y1, z1, x2, y2, z2, x3, y3, z3, m_str);
        } else {
            sprintf(buff, "{NOTHING}");
        }
        std::string ret(buff);
        return ret;
    }
};

std::ostream& operator<<(std::ostream& os, const MyObject& obj) {
    os << obj.to_string() << std::endl;
    return os;
}

class MyScene {
public:
    double background_color = 0.0;
    std::vector<MyObject> object_list;
    MyScene() {
        set_background_color();
    }
    friend std::ostream& operator<<(std::ostream& os, const MyScene& scene);
    void set_background_color(double _bgc=0) {
        background_color = _bgc;
    }
    void add_object(const MyObject & _obj) {
        // copy...?
        object_list.push_back(_obj);
    }
    double primary_ray_tracing(const cv::Mat & pt_start, const cv::Mat & unit_vec) const {
        assert(pt_start.size().width == 1);
        assert(pt_start.size().height == 3);
        assert(unit_vec.size().width == 1);
        assert(unit_vec.size().height == 3);
        int firstObjIdx = -1;
        double rayK = -1;
        internal_get_first_object(firstObjIdx, rayK, pt_start, unit_vec);
        if (firstObjIdx == -1) {
            return background_color;
        }

        cv::Mat light_xyz(3, 1, CV_64F, cv::Scalar(0.0));
        cv::Mat N(3, 1, CV_64F, cv::Scalar(0.0));
        cv::Mat V(3, 1, CV_64F, cv::Scalar(0.0));
        cv::Mat intersection(3, 1, CV_64F, cv::Scalar(0.0));

        if (object_list[firstObjIdx].t == MyObjectType::SPHERE) {
            // calculate V, N, intersection for sphere
            double Px = pt_start.at<double>(0,0);
            double Py = pt_start.at<double>(1,0);
            double Pz = pt_start.at<double>(2,0);
            double VxRaw = Px - object_list[firstObjIdx].x;
            double VyRaw = Py - object_list[firstObjIdx].y;
            double VzRaw = Pz - object_list[firstObjIdx].z;
            double InvVRawLen = 1./sqrt(VxRaw*VxRaw+VyRaw*VyRaw+VzRaw*VzRaw);
            V.at<double>(0,0) = VxRaw * InvVRawLen;
            V.at<double>(1,0) = VyRaw * InvVRawLen;
            V.at<double>(2,0) = VzRaw * InvVRawLen;
            intersection = pt_start + rayK * unit_vec;
            double NxRaw = intersection.at<double>(0,0) - object_list[firstObjIdx].x;
            double NyRaw = intersection.at<double>(1,0) - object_list[firstObjIdx].y;
            double NzRaw = intersection.at<double>(2,0) - object_list[firstObjIdx].z;
            double InvNRawLen = 1./sqrt(NxRaw*NxRaw+NyRaw*NyRaw+NzRaw*NzRaw);
            N.at<double>(0,0) = NxRaw * InvNRawLen;
            N.at<double>(1,0) = NyRaw * InvNRawLen;
            N.at<double>(2,0) = NzRaw * InvNRawLen;
        } else if (object_list[firstObjIdx].t == MyObjectType::PLANE) {
            // calculate V, N, intersection for plane
            V = (-1) * unit_vec;
            intersection = pt_start + rayK * unit_vec;
            double NxRaw = object_list[firstObjIdx].x;
            double NyRaw = object_list[firstObjIdx].y;
            double NzRaw = object_list[firstObjIdx].z;
            double InvNRawLen = 1./sqrt(NxRaw*NxRaw+NyRaw*NyRaw+NzRaw*NzRaw);
            N.at<double>(0,0) = NxRaw * InvNRawLen;
            N.at<double>(1,0) = NyRaw * InvNRawLen;
            N.at<double>(2,0) = NzRaw * InvNRawLen;
        } else if (object_list[firstObjIdx].t == MyObjectType::TRIANGLE) {
            // calculate V, N, intersection for triangle
            V = (-1) * unit_vec;
            intersection = pt_start + rayK * unit_vec;
            double nx, ny, nz;
            internal_triangle_normal(object_list[firstObjIdx].x1,
                                     object_list[firstObjIdx].y1,
                                     object_list[firstObjIdx].z1,
                                     object_list[firstObjIdx].x2,
                                     object_list[firstObjIdx].y2,
                                     object_list[firstObjIdx].z2,
                                     object_list[firstObjIdx].x3,
                                     object_list[firstObjIdx].y3,
                                     object_list[firstObjIdx].z3,
                                     nx,ny,nz);
            N.at<double>(0,0) = nx;
            N.at<double>(1,0) = ny;
            N.at<double>(2,0) = nz;
        }

        // return illumination
        double Ip = 0.0;
        // find all ambient light sum
        double ia = internal_get_ambient_light_sum();
        Ip += ia * object_list[firstObjIdx].ka;

        // calculate each light
        
        for (int i = 0; i < object_list.size(); i++) {
            if (object_list[i].t == MyObjectType::POINT_LIGHT) {
                light_xyz.at<double>(0,0) = object_list[i].x;
                light_xyz.at<double>(1,0) = object_list[i].y;
                light_xyz.at<double>(2,0) = object_list[i].z;
            } else if (object_list[i].t == MyObjectType::CYLINDER_LIGHT) {
                light_xyz.at<double>(0,0) = object_list[i].x;
                light_xyz.at<double>(1,0) = object_list[i].y;
                light_xyz.at<double>(2,0) = object_list[i].z;
            } else if (object_list[i].t == MyObjectType::CONE_LIGHT) {
                light_xyz.at<double>(0,0) = object_list[i].x;
                light_xyz.at<double>(1,0) = object_list[i].y;
                light_xyz.at<double>(2,0) = object_list[i].z;
            } else {
                continue; // not light source
            }
            cv::Mat light_to_obj_vec = intersection - light_xyz;
            double lovx = light_to_obj_vec.at<double>(0,0);
            double lovy = light_to_obj_vec.at<double>(1,0);
            double lovz = light_to_obj_vec.at<double>(2,0);
            double InvLenLov = 1./sqrt(lovx*lovx+lovy*lovy+lovz*lovz);
            light_to_obj_vec.at<double>(0,0) *= InvLenLov;
            light_to_obj_vec.at<double>(1,0) *= InvLenLov;
            light_to_obj_vec.at<double>(2,0) *= InvLenLov;
            int firstObjLightSeeIdx = -1;
            double RayKLightSee = -1;
            internal_get_first_object(firstObjLightSeeIdx, RayKLightSee, light_xyz, light_to_obj_vec);
            if (firstObjLightSeeIdx != firstObjIdx) {
                continue; // this light source can't see this point
            }

            if (object_list[i].t == MyObjectType::CYLINDER_LIGHT) {
                // TODO: check whether in cylinder
                double x_u = object_list[i].x_u;
                double y_u = object_list[i].y_u;
                double z_u = object_list[i].z_u;
                double itrsct_x = intersection.at<double>(0,0);
                double itrsct_y = intersection.at<double>(1,0);
                double itrsct_z = intersection.at<double>(2,0);
                double x = object_list[i].x;
                double y = object_list[i].y;
                double z = object_list[i].z;
                double delta_x = itrsct_x - x;
                double delta_y = itrsct_y - y;
                double delta_z = itrsct_z - z;
                // k = np.inner(obj.light_unit_vec, intersection-obj.light_start)
                double k = x_u*delta_x+y_u*delta_y+z_u*delta_z;
                if (k < 0) {
                    continue; // back side of light
                }
                double dx = k*x_u - delta_x;
                double dy = k*y_u - delta_y;
                double dz = k*z_u - delta_z;
                double len_d_sq = dx*dx + dy*dy + dz*dz;
                //d = (obj.light_start + k*obj.light_unit_vec - intersection)
                if (len_d_sq > (object_list[i].r*object_list[i].r)) {
                    continue; // out of cylinder
                }
            }

            if (object_list[i].t == MyObjectType::CONE_LIGHT) {
                // TODO: check whether in cylinder
                double x_u = object_list[i].x_u;
                double y_u = object_list[i].y_u;
                double z_u = object_list[i].z_u;
                double itrsct_x = intersection.at<double>(0,0);
                double itrsct_y = intersection.at<double>(1,0);
                double itrsct_z = intersection.at<double>(2,0);
                double x = object_list[i].x;
                double y = object_list[i].y;
                double z = object_list[i].z;
                double delta_x = itrsct_x - x;
                double delta_y = itrsct_y - y;
                double delta_z = itrsct_z - z;
                double delta_len = sqrt(delta_x*delta_x+delta_y*delta_y+delta_z*delta_z);
                // k = np.inner(obj.light_unit_vec, intersection-obj.light_start)
                double k = x_u*delta_x+y_u*delta_y+z_u*delta_z;
                if (k < object_list[i].min_inner * delta_len) {
                    continue; // out of cone
                }
            }


            if (object_list[firstObjIdx].t == MyObjectType::SPHERE) {
                // if the same object, check whether the same point
                cv::Mat LtSS = intersection - light_xyz;
                cv::Mat first_xyz(3, 1, CV_64F, cv::Scalar(0.0));
                first_xyz.at<double>(0,0) = object_list[firstObjIdx].x;
                first_xyz.at<double>(1,0) = object_list[firstObjIdx].y;
                first_xyz.at<double>(2,0) = object_list[firstObjIdx].z;
                cv::Mat LtSC = first_xyz - light_xyz;

                cv::Mat LtSSLtSS = LtSS.t() * LtSS;
                double InnerLtSSLtSS = LtSSLtSS.at<double>(0,0);
                cv::Mat LtSCLtSS = LtSC.t() * LtSS;
                double InnerLtSCLtSS = LtSCLtSS.at<double>(0,0);

                if (InnerLtSSLtSS > InnerLtSCLtSS) {
                    continue; 
                }
            }

            cv::Mat Lm = light_xyz - intersection;
            double Lmx = Lm.at<double>(0,0);
            double Lmy = Lm.at<double>(1,0);
            double Lmz = Lm.at<double>(2,0);
            double InvLenLm = 1./sqrt(Lmx*Lmx+Lmy*Lmy+Lmz*Lmz);
            Lm.at<double>(0,0) *= InvLenLm;
            Lm.at<double>(1,0) *= InvLenLm;
            Lm.at<double>(2,0) *= InvLenLm;
            cv::Mat LmN = Lm.t() * N;
            double InnerLmN = LmN.at<double>(0,0);
            cv::Mat Rm = 2*InnerLmN*N - Lm;
            cv::Mat RmV = Rm.t() * V;
            double InnerRmV = RmV.at<double>(0,0);
            double InnerRmVAbs = (InnerRmV > 0) ? InnerRmV : -InnerRmV;
            double imd = object_list[i].i_diffused;
            double ims = object_list[i].i_specular;
            if (object_list[firstObjIdx].t == MyObjectType::PLANE ||
                object_list[firstObjIdx].t == MyObjectType::TRIANGLE) {
                double InnerLmNAbs = (InnerLmN > 0) ? InnerLmN : -InnerLmN;
                Ip += object_list[firstObjIdx].kd*InnerLmNAbs*imd; // diffused
                Ip += object_list[firstObjIdx].ks*pow(InnerRmVAbs,object_list[firstObjIdx].alpha)*ims; // specular
            } else {
                Ip += object_list[firstObjIdx].kd*InnerLmN*imd; // diffused
                Ip += object_list[firstObjIdx].ks*pow(InnerRmVAbs,object_list[firstObjIdx].alpha)*ims; // specular
            }
        }
        return Ip;
    }

    double internal_get_ambient_light_sum() const {
        double ia = 0;
        for (int i = 0; i < object_list.size(); i++) {
            if (object_list[i].t == MyObjectType::AMBIENT_LIGHT) {
                ia += object_list[i].illumination;
            }
        }
        return ia;
    }


    void internal_get_first_object(int & firstObjIdx, double & rayK, const cv::Mat & pt_start, const cv::Mat & unit_vec) const {
        double tmp_min_k = INFINITY;
        double tmp_k = INFINITY;
        for (int i = 0; i < object_list.size(); i++) {
            if (object_list[i].t == MyObjectType::SPHERE) {
                if (internal_intersect_sphere(object_list[i].x,
                                              object_list[i].y,
                                              object_list[i].z,
                                              object_list[i].r,
                                              pt_start,
                                              unit_vec,
                                              tmp_k)) {
                    if (tmp_k < tmp_min_k) {
                        firstObjIdx = i;
                        tmp_min_k = tmp_k;
                    }
                }
            } else if (object_list[i].t == MyObjectType::PLANE) {
                if (internal_intersect_plane(object_list[i].x,
                                             object_list[i].y,
                                             object_list[i].z,
                                             object_list[i].xyz_sum,
                                             pt_start,
                                             unit_vec,
                                             tmp_k)) {
                    if (tmp_k < tmp_min_k) {
                        firstObjIdx = i;
                        tmp_min_k = tmp_k;
                    }
                }
            } else if (object_list[i].t == MyObjectType::TRIANGLE) {
                if (internal_intersect_triangle(object_list[i].x1,
                                                object_list[i].y1,
                                                object_list[i].z1,
                                                object_list[i].x2,
                                                object_list[i].y2,
                                                object_list[i].z2,
                                                object_list[i].x3,
                                                object_list[i].y3,
                                                object_list[i].z3,
                                                pt_start,
                                                unit_vec,
                                                tmp_k)) {
                    if (tmp_k < tmp_min_k) {
                        firstObjIdx = i;
                        tmp_min_k = tmp_k;
                    }
                }
            }
        }
        rayK = tmp_min_k;
    }

    int internal_three_point_order(double _x1,
                                      double _y1,
                                      double _x2,
                                      double _y2,
                                      double _x3,
                                      double _y3) const {
        double raw = _x1*_y2+_x2*_y3+_x3*_y1-_x1*_y3-_x2*_y1-_x3*_y2;
        return (raw > 0) ? 1 : -1; // return sign
    }

    void internal_triangle_normal(double _x1,
                                  double _y1,
                                  double _z1,
                                  double _x2,
                                  double _y2,
                                  double _z2,
                                  double _x3,
                                  double _y3,
                                  double _z3,
                                  double & nx,
                                  double & ny,
                                  double & nz) const {
        double v12x = _x2 - _x1;
        double v12y = _y2 - _y1;
        double v12z = _z2 - _z1;
        double v13x = _x3 - _x1;
        double v13y = _y3 - _y1;
        double v13z = _z3 - _z1;
        double v23x = _x3 - _x2;
        double v23y = _y3 - _y2;
        double v23z = _z3 - _z2;
        double len_v12_sq = v12x*v12x+v12y*v12y+v12z*v12z;
        double len_v13_sq = v13x*v13x+v13y*v13y+v13z*v13z;
        double len_v23_sq = v23x*v23x+v23y*v23y+v23z*v23z;
        assert(len_v12_sq > 0);
        assert(len_v13_sq > 0);
        assert(len_v23_sq > 0);
        double inv_len_v12 = 1./sqrt(len_v12_sq);
        double inv_len_v13 = 1./sqrt(len_v13_sq);
        double inv_len_v23 = 1./sqrt(len_v23_sq);
        double uv12x = v12x * inv_len_v12;
        double uv12y = v12y * inv_len_v12;
        double uv12z = v12z * inv_len_v12;
        double uv13x = v13x * inv_len_v13;
        double uv13y = v13y * inv_len_v13;
        double uv13z = v13z * inv_len_v13;
        double uv23x = v23x * inv_len_v23;
        double uv23y = v23y * inv_len_v23;
        double uv23z = v23z * inv_len_v23;
        double AbsInneruv1213 = abs(uv12x*uv13x+uv12y*uv13y+uv12z*uv13z);
        double AbsInneruv1223 = abs(uv12x*uv23x+uv12y*uv23y+uv12z*uv23z);
        double AbsInneruv1323 = abs(uv13x*uv23x+uv13y*uv23y+uv13z*uv23z);
        assert(AbsInneruv1213 < 1);
        assert(AbsInneruv1223 < 1);
        assert(AbsInneruv1323 < 1);
        // use the minumum one to calculate normal vector for numerically stablility
        if ((AbsInneruv1213 < AbsInneruv1223) && (AbsInneruv1213 < AbsInneruv1323)) {
            // use uv12 and uv13
            nx = uv12y*uv13z - uv12z*uv13y;
            ny = uv12z*uv13x - uv12x*uv13z;
            nz = uv12x*uv13y - uv12y*uv13x;
        } else if (AbsInneruv1223 < AbsInneruv1323) {
            // use uv12 and uv23
            nx = uv12y*uv23z - uv12z*uv23y;
            ny = uv12z*uv23x - uv12x*uv23z;
            nz = uv12x*uv23y - uv12y*uv23x;
        } else {
            // ise uv13 and uv23
            nx = uv13y*uv23z - uv13z*uv23y;
            ny = uv13z*uv23x - uv13x*uv23z;
            nz = uv13x*uv23y - uv13y*uv23x;
        }
    }

    bool internal_intersect_triangle(double _x1,
                                     double _y1,
                                     double _z1,
                                     double _x2,
                                     double _y2,
                                     double _z2,
                                     double _x3,
                                     double _y3,
                                     double _z3,
                                     cv::Mat pt_start,
                                     cv::Mat unit_vec,
                                     double & tmp_k) const {
        double nx, ny, nz;
        internal_triangle_normal(_x1,_y1,_z1,_x2,_y2,_z2,_x3,_y3,_z3,nx,ny,nz);
        double v12x = _x2 - _x1;
        double v12y = _y2 - _y1;
        double v12z = _z2 - _z1;
        double len_v12_sq = v12x*v12x+v12y*v12y+v12z*v12z;
        double inv_len_v12 = 1./sqrt(len_v12_sq);
        double uv12x = v12x * inv_len_v12;
        double uv12y = v12y * inv_len_v12;
        double uv12z = v12z * inv_len_v12;
        double tmp_xyz_sum = nx*_x1+ny*_y1+nz*_z1;
        bool ret = internal_intersect_plane(nx, ny, nz, tmp_xyz_sum, pt_start, unit_vec, tmp_k);
        if (!ret) {
            tmp_k = INFINITY;
            return false;
        }
        cv::Mat pd = pt_start + tmp_k*unit_vec;
        double pdx = pd.at<double>(0,0);
        double pdy = pd.at<double>(1,0);
        double pdz = pd.at<double>(2,0);

        // third_axis = np.cross(uv12, tmp_normal)
        double tax = uv12y*nz - uv12z*ny;
        double tay = uv12z*nx - uv12x*nz;
        double taz = uv12x*ny - uv12y*nx;
        //p1 = np.array([x1,y1,z1], dtype=np.float64)
        //p2 = np.array([x2,y2,z2], dtype=np.float64)
        //p3 = np.array([x3,y3,z3], dtype=np.float64)
        //use uv12, third_axis, tmp_normal to represent p1, p2, p3, pd
        double p1_tx = _x1*uv12x + _y1*uv12y + _z1*uv12z;
        double p1_ty = _x1*  tax + _y1*  tay + _z1*  taz;
        double p2_tx = _x2*uv12x + _y2*uv12y + _z2*uv12z;
        double p2_ty = _x2*  tax + _y2*  tay + _z2*  taz;
        double p3_tx = _x3*uv12x + _y3*uv12y + _z3*uv12z;
        double p3_ty = _x3*  tax + _y3*  tay + _z3*  taz;
        double pd_tx = pdx*uv12x + pdy*uv12y + pdz*uv12z;
        double pd_ty = pdx*  tax + pdy*  tay + pdz*  taz;
        int o123 = internal_three_point_order(p1_tx,p1_ty,p2_tx,p2_ty,p3_tx,p3_ty);
        int o12d = internal_three_point_order(p1_tx,p1_ty,p2_tx,p2_ty,pd_tx,pd_ty);
        int o23d = internal_three_point_order(p2_tx,p2_ty,p3_tx,p3_ty,pd_tx,pd_ty);
        int o31d = internal_three_point_order(p3_tx,p3_ty,p1_tx,p1_ty,pd_tx,pd_ty);
        if (o123==o12d && o123==o23d && o123==o31d) {
            return true;
        }
        tmp_k = INFINITY;
        return false;
    }

    bool internal_intersect_sphere(double _Px, // sphere_center_x
                                   double _Py, // sphere_center_y
                                   double _Pz, // sphere_center_z
                                   double _r, //sphere_radius
                                   const cv::Mat & pt_start,
                                   const cv::Mat & unit_vec,
                                   double & tmp_k) const {
        double _Ax = pt_start.at<double>(0,0);
        double _Ay = pt_start.at<double>(1,0);
        double _Az = pt_start.at<double>(2,0);
        double _Bx = unit_vec.at<double>(0,0);
        double _By = unit_vec.at<double>(1,0);
        double _Bz = unit_vec.at<double>(2,0);
        double _k = _Bx*(_Px-_Ax)+_By*(_Py-_Ay)+_Bz*(_Pz-_Az);
        double _dx = (_Ax + _k*_Bx - _Px);
        double _dy = (_Ay + _k*_By - _Py);
        double _dz = (_Az + _k*_Bz - _Pz);
        double _d_norm_sq = _dx*_dx+_dy*_dy+_dz*_dz;
        // std::cout << _Ax << "," << _Ay << "," << _Az << "\n";
        // std::cout << _Bx << "," << _By << "," << _Bz << "\n";
        // std::cout << _k << "\n";
        // std::cout << _d_norm_sq << "\n";
        double _r_sq = _r*_r;
        if (_d_norm_sq > _r_sq) {
            tmp_k = INFINITY;
            return false;
        }
        double k_delta = sqrt(_r_sq-_d_norm_sq);
        if ((_k + k_delta) < 0) {
            tmp_k = INFINITY;
            return false;            
        }
        if ((_k - k_delta) > 0) {
            tmp_k = _k - k_delta;         
        } else {
            tmp_k = _k + k_delta;
        }
        return true;
    }

    bool internal_intersect_plane(double _x_coeff,
                                  double _y_coeff,
                                  double _z_coeff,
                                  double _xyz_sum,
                                  cv::Mat pt_start,
                                  cv::Mat unit_vec,
                                  double & tmp_k) const {
        double _Ax = pt_start.at<double>(0,0);
        double _Ay = pt_start.at<double>(1,0);
        double _Az = pt_start.at<double>(2,0);
        double _Bx = unit_vec.at<double>(0,0);
        double _By = unit_vec.at<double>(1,0);
        double _Bz = unit_vec.at<double>(2,0);
        double Inner_plane_normal_unit_vec = _x_coeff*_Bx+_y_coeff*_By+_z_coeff*_Bz;
        double Inner_plane_normal_pt_start = _x_coeff*_Ax+_y_coeff*_Ay+_z_coeff*_Az;
        if (Inner_plane_normal_unit_vec==0) {
            if (Inner_plane_normal_pt_start==_xyz_sum) {
                tmp_k = 0.0;
                return true;
            } else {
                tmp_k = INFINITY;
                return false;
            }
        }
        tmp_k = (_xyz_sum - Inner_plane_normal_pt_start) / Inner_plane_normal_unit_vec;
        return true;
    }
/*
        def internal_intersect_plane(self, x_coeff, y_coeff, z_coeff, xyz_sum, pt_start, unit_vec):
        plane_normal = np.array([x_coeff,y_coeff,z_coeff], dtype=np.float64)
        if np.inner(plane_normal,unit_vec) == 0:
            if np.inner(pt_start,plane_normal) == xyz_sum:
                return True, 0.
            else:
                return False, 0.
        k = (xyz_sum - np.inner(plane_normal,pt_start)) / np.inner(plane_normal,unit_vec)
        return True, k
*/
    std::string to_string() const {
        std::string ret("");
        std::string tmpnewline("\n");
        for (int i = 0; i < object_list.size(); i++) {
            ret += object_list[i].to_string() + tmpnewline;
        }
        return ret;
    }
};

std::ostream& operator<<(std::ostream& os, const MyScene& scene) {
    os << scene.to_string() << std::endl;
    return os;
}

class MyRenderer {
public:
    cv::Mat K;
    cv::Mat RT;
    cv::Mat dist;
    int width;
    int height;
    MyRenderer(const struct MyCamModel & _cm, int _w, int _h) {
        K = _cm.K.clone();
        RT = _cm.RT.clone();
        dist = _cm.dist.clone();
        width = _w;
        height = _h;
        assert(_w > 0);
        assert(_h > 0);
    }
    void render_primary_ray_tracing(const MyScene & scene, cv::Mat & ret_64f) const {
        std::cout << "rendering..." << std::endl;
        // std::cout << "K = " << std::endl << K << std::endl;
        double fx = K.at<double>(0,0);
        double fy = K.at<double>(1,1);
        double cx = K.at<double>(0,2);
        double cy = K.at<double>(1,2);
        cv::Mat R(3, 3, CV_64F, cv::Scalar(0.0));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R.at<double>(i,j) = RT.at<double>(i,j);
            }
        }
        cv::Mat T(3, 1, CV_64F, cv::Scalar(0.0));
        for (int i = 0; i < 3; i++) {
            T.at<double>(i,0) = RT.at<double>(i,3);
        }
        // std::cout << "RT = " << std::endl << RT << std::endl;
        // std::cout << "R = " << std::endl << R << std::endl;
        // std::cout << "T = " << std::endl << T << std::endl;
        cv::Mat cam_center_world(3, 1, CV_64F, cv::Scalar(0.0));
        cv::Mat RTranspose = R.t();
        cam_center_world = (-1) * R.t() * T;
        // std::cout << "cam_center_world = " << std::endl << cam_center_world << std::endl;

        cv::Mat img_64f(height, width, CV_64F, cv::Scalar(0.0));

        cv::Mat pt2d(1, 2, CV_64F, cv::Scalar(0.0));
        // cv::Mat pt2d_corrected(1, 2, CV_64F, cv::Scalar(0.0));
        cv::Mat trace_loc_cam(1, 2, CV_64F, cv::Scalar(0.0));
        cv::Mat trace_loc_camz(3, 1, CV_64F, cv::Scalar(0.0));
        cv::Mat trace_loc_world;
        cv::Mat unit_vec;
        // for (int x = 0; x < 1; x++) {
        // for (int x = 190; x < 191; x++) {
        for (int x = 0; x < width; x++) {
            if (x % 200 == 0) {
                std::cout << "Progress: " << float(100.*x/width) << "% (" << x << "/" << width << ")\n";
            }
            // for (int y = 0; y < 1; y++) {
            for (int y = 0; y < height; y++) {
            // for (int y = 559; y < 560; y++) {
                pt2d.at<double>(0,0) = x;
                pt2d.at<double>(0,1) = y;
                cv::undistortPoints(pt2d, trace_loc_cam, K, dist);
                trace_loc_camz.at<double>(0,0) = trace_loc_cam.at<double>(0,0);
                trace_loc_camz.at<double>(1,0) = trace_loc_cam.at<double>(0,1);
                trace_loc_camz.at<double>(2,0) = 1.0;
                unit_vec = R.t() * trace_loc_camz;
                double unit_vec_len = 0;
                for (int i_unit_vec_dim = 0; i_unit_vec_dim < 3; i_unit_vec_dim++) {
                    double tmp_dim = unit_vec.at<double>(i_unit_vec_dim,0);
                    unit_vec_len += tmp_dim * tmp_dim;
                }
                unit_vec_len = sqrt(unit_vec_len);
                unit_vec /= cv::Scalar(unit_vec_len);
                double color = scene.primary_ray_tracing(cam_center_world, unit_vec);
                img_64f.at<double>(y,x) = color;
                // std::cout << "pt_start=" << std::endl << cam_center_world << std::endl;
                // std::cout << "unit_vec=" << std::endl << unit_vec << std::endl;
                // trace_loc_world = R.t() * trace_loc_camz + cam_center_world;
                // std::cout << trace_loc_cam << std::endl;
                // std::cout << trace_loc_world << std::endl;
                // cv::undistortPoints(pt2d, pt2d_corrected, K, dist, cv::noArray(), K);
                // std::cout << pt2d << std::endl;
                // std::cout << pt2d_corrected << std::endl;
                // cv2.undistortPoints(pt2d.reshape(1,1,2), self.K, self.dist, None, self.K)
            }
        }
        ret_64f = img_64f.clone();
    }
};

void GetCamParamFromNpz(const char* cam_model_npz,
                        struct MyCamModel & LCam, 
                        struct MyCamModel & RCam) {

    cnpy::NpyArray arr;
    // distL
    arr = cnpy::npz_load(cam_model_npz,"distL");
    assert(arr.word_size == sizeof(double));
    assert(arr.shape.size() == 2);
    assert(arr.shape[0] == 1);
    cv::Mat distL(arr.shape[0], arr.shape[1], CV_64F, arr.data<double>());
    LCam.dist = distL.clone();
    // distR
    arr = cnpy::npz_load(cam_model_npz,"distR");
    assert(arr.word_size == sizeof(double));
    assert(arr.shape.size() == 2);
    assert(arr.shape[0] == 1);
    cv::Mat distR(arr.shape[0], arr.shape[1], CV_64F, arr.data<double>());
    RCam.dist = distR.clone();
    // L_K
    arr = cnpy::npz_load(cam_model_npz,"intrinsicL");
    assert(arr.word_size == sizeof(double));
    assert(arr.shape.size() == 2);
    assert(arr.shape[0] == 3);
    assert(arr.shape[1] == 3);
    cv::Mat L_K(arr.shape[0], arr.shape[1], CV_64F, arr.data<double>());
    LCam.K = L_K.clone();
    // R_K
    arr = cnpy::npz_load(cam_model_npz,"intrinsicR");
    assert(arr.word_size == sizeof(double));
    assert(arr.shape.size() == 2);
    assert(arr.shape[0] == 3);
    assert(arr.shape[1] == 3);
    cv::Mat R_K(arr.shape[0], arr.shape[1], CV_64F, arr.data<double>());
    RCam.K = R_K.clone();
    // L_RT
    arr = cnpy::npz_load(cam_model_npz,"L_RT");
    assert(arr.word_size == sizeof(double));
    assert(arr.shape.size() == 2);
    assert(arr.shape[0] == 3);
    assert(arr.shape[1] == 4);
    cv::Mat L_RT(arr.shape[0], arr.shape[1], CV_64F, arr.data<double>());
    LCam.RT = L_RT.clone();
    // R_RT
    arr = cnpy::npz_load(cam_model_npz,"R_RT");
    assert(arr.word_size == sizeof(double));
    assert(arr.shape.size() == 2);
    assert(arr.shape[0] == 3);
    assert(arr.shape[1] == 4);
    cv::Mat R_RT(arr.shape[0], arr.shape[1], CV_64F, arr.data<double>());
    RCam.RT = R_RT.clone();
/*
    std::cout << "LCam.dist = " << std::endl << " "  << LCam.dist << std::endl;
    std::cout << "RCam.dist = " << std::endl << " "  << RCam.dist << std::endl;
    std::cout << "LCam.K = " << std::endl << " "  << LCam.K << std::endl;
    std::cout << "RCam.K = " << std::endl << " "  << RCam.K << std::endl;
    std::cout << "LCam.RT = " << std::endl << " "  << LCam.RT << std::endl;
    std::cout << "RCam.RT = " << std::endl << " "  << RCam.RT << std::endl;
*/
    return;
}

# endif // SIMPLE_RAY_TRACE_HEADER
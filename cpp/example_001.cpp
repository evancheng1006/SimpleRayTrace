

#include "cnpy.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>

#include "SimpleRayTrace.h"

int main(int argc, char* argv[]) {
    struct MyCamModel LCam, RCam;
    int width = 2048;
    int height = 2048;
    GetCamParamFromNpz("camParam-1612675835485.npz", LCam, RCam);

    MyRenderer Lrd(LCam, width, height);
    MyRenderer Rrd(RCam, width, height);

    MyScene sc;

    MyObject ambl;
    ambl.set_ambient_light(0.5);
    MyObject ptl;
    ptl.set_point_light(30.,-30.,-10.,1.,1.);
    MyObject cyl;
    cyl.set_cylinder_light(9, 9, -200, 0, 0, 1, 0.1, 50.0, 0.0);
    MyObject cne;
    cne.set_cone_light(7, 7, -80, 0, 0, 1, 1, 50.0, 50.0);

    MyObject sph;
    sph.set_sphere(8.,8.,0.,5.);
    sph.set_phong_material(0.5, 0.2, 0.2, 40);
    MyObject pln;
    pln.set_plane(0.,0.,1.,1.);
    pln.set_phong_material(0.0, 0.1, 0.05, 1);

    // cube
    MyObject tri1, tri2, tri3, tri4;
    MyObject tri5, tri6, tri7, tri8;
    MyObject tri9, tri10, tri11, tri12;
    tri1.set_triangle(0,0,0,0,5,0,5,0,0);
    tri2.set_triangle(5,5,0,0,5,0,5,0,0);
    tri3.set_triangle(0,0,5,0,5,5,5,0,5);
    tri4.set_triangle(5,5,5,0,5,5,5,0,5);
    tri5.set_triangle(0,0,0,0,5,0,0,0,5);
    tri6.set_triangle(0,5,5,0,5,0,0,0,5);
    tri7.set_triangle(5,0,0,5,5,0,5,0,5);
    tri8.set_triangle(5,5,5,5,5,0,5,0,5);
    tri9.set_triangle(0,0,0,0,0,5,5,0,0);
    tri10.set_triangle(5,0,5,0,0,5,5,0,0);
    tri11.set_triangle(0,5,0,0,5,5,5,5,0);
    tri12.set_triangle(5,5,5,0,5,5,5,5,0);

    tri1.set_phong_material(0.2,0.2,0.2,40);
    tri2.set_phong_material(0.2,0.2,0.2,40);
    tri3.set_phong_material(0.2,0.2,0.2,40);
    tri4.set_phong_material(0.2,0.2,0.2,40);
    tri5.set_phong_material(0.2,0.2,0.2,40);
    tri6.set_phong_material(0.2,0.2,0.2,40);
    tri7.set_phong_material(0.2,0.2,0.2,40);
    tri8.set_phong_material(0.2,0.2,0.2,40);
    tri9.set_phong_material(0.2,0.2,0.2,40);
    tri10.set_phong_material(0.2,0.2,0.2,40);
    tri11.set_phong_material(0.2,0.2,0.2,40);
    tri12.set_phong_material(0.2,0.2,0.2,40);


    sc.add_object(ambl);
    sc.add_object(ptl);
    sc.add_object(cyl);
    sc.add_object(cne);
    sc.add_object(sph);
    sc.add_object(pln);
    sc.add_object(tri1);
    sc.add_object(tri2);
    sc.add_object(tri3);
    sc.add_object(tri4);
    sc.add_object(tri5);
    sc.add_object(tri6);
    sc.add_object(tri7);
    sc.add_object(tri8);
    sc.add_object(tri9);
    sc.add_object(tri10);
    sc.add_object(tri11);
    sc.add_object(tri12);

    std::cout << "scene=\n" << sc;
    std::ofstream f_scene_txt("scene.txt");
    f_scene_txt << sc.to_string();
    f_scene_txt.close();
    std::cout << "saving scene to scene.txt" << std::endl;

    cv::Mat Limg, Rimg;
    Lrd.render_primary_ray_tracing(sc, Limg);
    Rrd.render_primary_ray_tracing(sc, Rimg);
    cv::FileStorage Lxml("Limg_64F.xml", cv::FileStorage::WRITE);
    Lxml << "img" << Limg;
    cv::FileStorage Rxml("Rimg_64F.xml", cv::FileStorage::WRITE);
    Rxml << "img" << Rimg;
    std::cout << "saving raw output to Limg_64F.xml" << std::endl;
    std::cout << "saving raw output to Rimg_64F.xml" << std::endl;

    Limg.convertTo(Limg, CV_8UC1, 255.0); // opencv has saturate_cast
    Rimg.convertTo(Rimg, CV_8UC1, 255.0); // opencv has saturate_cast
    cv::imwrite("Limg.png", Limg);
    cv::imwrite("Rimg.png", Rimg);

    std::cout << "saving output to Limg.png" << std::endl;
    std::cout << "saving output to Rimg.png" << std::endl;

    cv::Mat LtmpImg, RtmpImg;
    cv::resize(Rimg, RtmpImg, cv::Size(), 0.4, 0.4);
    cv::namedWindow("Rimg", cv::WINDOW_AUTOSIZE);
    cv::imshow("Rimg", RtmpImg);
    cv::resize(Limg, LtmpImg, cv::Size(), 0.4, 0.4);
    cv::namedWindow("Limg", cv::WINDOW_AUTOSIZE);
    cv::imshow("Limg", LtmpImg);
    cv::waitKey(0);

    return 0;
}

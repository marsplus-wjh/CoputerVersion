//
// Created by 王家辉 on 2019-02-03.
//

#ifndef CVA2_M_DESCRIPTOR_H
#define CVA2_M_DESCRIPTOR_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "vector"
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace cv;
using namespace std;

class Descriptor {
public:
    Mat descriptor(Mat& img, vector<KeyPoint>& keypoints);
    Mat getDescriptor(Mat& src, KeyPoint keyPoint, Mat& Kernel_Gau);
    Mat getPatch(Mat& img, KeyPoint keyPoint);
    void guassian(int sizex, int sizey, double sigma, Mat& kernel);
    Mat Gaussian_x(int ksize, float sigma);
    Mat Gaussian_y(int ksize, float sigma);
    int Theta(Mat &patch_mag, Mat &patch_dir);
    bool isExtremum(int x,int y,const vector<Mat>& dog_pyr,int index);
};


#endif //CVA2_M_DESCRIPTOR_H

//
// Created by 王家辉 on 2019-03-08.
//

#ifndef CVPRO_HARISCONER_H
#define CVPRO_HARISCONER_H

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

class HarisConer {
public:
    void detector(Mat& src, Mat& imgDst, double qualityLevel, vector<cv::Point2f> &points);
    void guassian(int sizex, int sizey, double sigma, Mat& kernel);
    bool checkLocalMaximum(Mat &c, int x, int y);
    void drawOnImage(Mat &image, const vector<cv::Point2f> &points);
};


#endif //CVPRO_HARISCONER_H

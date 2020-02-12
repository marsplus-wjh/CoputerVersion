//
// Created by 王家辉 on 2019-02-01.
//

#ifndef CVA2_M_HARISCONER_H
#define CVA2_M_HARISCONER_H

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
    void detect(const cv::Mat& image);
    void drawOnImage(Mat &image, const vector<cv::Point2f> &points);
};


#endif //CVA2_M_HARISCONER_H

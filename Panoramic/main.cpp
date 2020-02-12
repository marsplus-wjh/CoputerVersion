//
// Created by 王家辉 on 2019-03-08.
//

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "vector"
#include "HarisConer.h"
#include "Descriptor.h"
#include "Matcher.h"

using namespace cv;
using namespace std;

int main() {
    //const char* imagename1 = "project_images/Boxes.png";

    const char* imagename1 = "project_images/Rainier1.png";
    const char* imagename2 = "project_images/Rainier2.png";
//    const char* imagename1 = "building12.png";
//    const char* imagename2 = "project_images/self/library3.jpeg";
    const char* imagename3 = "project_images/Rainier3.png";
    const char* imagename4 = "project_images/Rainier4.png";
    const char* imagename5 = "project_images/Rainier5.png";
    const char* imagename6 = "project_images/Rainier6.png";

    //pyrDown(self_1, self_1);

//    const char* imagename1 = "project_images/self/self_1.png";
//    const char* imagename2 = "project_images/self/self_2.png";
//    const char* imagename3 = "project_images/self/self_3.png";
//    const char* imagename4 = "project_images/self/self_4.png";
//    const char* imagename5 = "project_images/self/self_5.png";
//    const char* imagename6 = "project_images/self/self_6.png";

//    const char* imagename1 = "project_images/self/lib1.jpeg";
//    const char* imagename2 = "project_images/self/lib2.jpeg";
//    const char* imagename3 = "project_images/self/lib3.jpeg";
//    const char* imagename4 = "self_lib12.png";

    const char* imagename56 = "56.png";
    const char* imagename564 = "564.png";
    const char* imagename5642 = "5642.png";
    const char* imagename56423 = "56423.png";

//    const char* imagename_self12 = "self_12.png";
//    const char* imagename_self123 = "self_123.png";
//    const char* imagename_self1234 = "self_1234.png";
//    const char* imagename_self12345 = "self_12345.png";
//
//    const char* imagename_self45 = "self_45.png";

    Mat image1_C = imread(imagename1, 1);
    Mat image2_C = imread(imagename2, 1);
    Mat image3_C = imread(imagename3, 1);
    Mat image4_C = imread(imagename4, 1);
    Mat image5_C = imread(imagename5, 1);
    Mat image6_C = imread(imagename6, 1);

    Mat image56_C = imread(imagename56, 1);
    Mat image564_C = imread(imagename564, 1);
    Mat image5642_C = imread(imagename5642, 1);
    Mat image56423_C = imread(imagename56423, 1);

//    Mat imageSelf12_C = imread(imagename_self12, 1);
//    Mat imageSelf123_C = imread(imagename_self123, 1);
//    Mat imageSelf1234_C = imread(imagename_self1234, 1);
//    Mat imageSelf12345_C = imread(imagename_self12345, 1);
//
//    Mat imageSelf45_C = imread(imagename_self45, 1);


    //Phase 1
    HarisConer harisConer;
    Mat dst;
    vector<cv::Point2f> featurePts;
    harisConer.detector(image1_C, dst, 0.2, featurePts);

    Mat dst2;
    vector<cv::Point2f> featurePts2;
    harisConer.detector(image2_C, dst2, 0.2, featurePts2);

    //Phase 2
    vector<KeyPoint> keyPoint1;
    KeyPoint::convert(featurePts, keyPoint1);
    Descriptor Descriptor;
    Mat descriptors_1 = Descriptor.descriptor(image1_C, keyPoint1);
    Mat image1_Corner;
    drawKeypoints(image1_C, keyPoint1, image1_Corner);

    vector<KeyPoint> keyPoint2;
    KeyPoint::convert(featurePts2, keyPoint2);
    Mat descriptors_2 = Descriptor.descriptor(image2_C, keyPoint2);
    Mat image2_Corner;
    drawKeypoints(image2_C, keyPoint2, image2_Corner);
    cout << keyPoint1.size()<<endl;
    cout << keyPoint2.size()<<endl;


    Matcher matcher(descriptors_1, descriptors_2);
    Mat stitch1;
    matcher.run(image1_C, image2_C, keyPoint1, keyPoint2, stitch1);
    imshow("stitched", stitch1);
    //imwrite("building12.png", stitch1);

    waitKey(0);
    return 0;
}
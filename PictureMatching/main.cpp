//
// Created by 王家辉 on 2019-02-01.
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
    const char* imagename1 = "assignment2_sample_images/yosemite/Yosemite1.jpg";
    const char* imagename2 = "assignment2_sample_images/yosemite/Yosemite2.jpg";
//    const char* imagename2 = "assignment2_sample_images/yosemite/image2_rotate.jpg";
//    const char* imagename2 = "assignment2_sample_images/yosemite/Yosemite2_con.jpg";
//    const char* imagename2 = "assignment2_sample_images/yosemite/image2_per.jpeg";
//    const char* imagename2 = "assignment2_sample_images/yosemite/image2_scale.jpeg";


    Mat image1_C = imread(imagename1, 1);
    Mat image2_C = imread(imagename2, 1);
    if(image1_C.empty())
    {
        fprintf(stderr, "Can not load image %s\n", imagename1);
        return -1;
    }
    if(image2_C.empty())
    {
        fprintf(stderr, "Can not load image %s\n", imagename2);
        return -1;
    }

    //Phase 1
    HarisConer harisConer;
    Mat dst;
    vector<cv::Point2f> featurePts;
    harisConer.detector(image1_C, dst, 0.35, featurePts);

    Mat dst2;
    vector<cv::Point2f> featurePts2;
    harisConer.detector(image2_C, dst2, 0.3, featurePts2);
    imshow("Harris_Points", dst);

    //Phase 2
    vector<KeyPoint> keyPoint1;
    KeyPoint::convert(featurePts, keyPoint1);
    Descriptor Descriptor;
    Mat descriptors_1 = Descriptor.descriptor(image1_C, keyPoint1);
    //cout<< descriptors_1<< endl;
    drawKeypoints(image1_C, keyPoint1, image1_C);
    imshow("Corner", image1_C);
//    cout << keyPoint1.size()<<endl;

    vector<KeyPoint> keyPoint2;
    KeyPoint::convert(featurePts2, keyPoint2);
    Mat descriptors_2 = Descriptor.descriptor(image2_C, keyPoint2);
    //cout<< descriptors_2<< endl;
    drawKeypoints(image2_C, keyPoint2, image2_C);
    imshow("Corner2", image2_C);
    cout << keyPoint2.size()<<endl;
    //cout<< descriptors_1<<endl;

    Matcher matcher(descriptors_1, descriptors_2);
    matcher.run(image1_C, image2_C, keyPoint1, keyPoint2);

    //code from web for testing
    xfeatures2d::SiftFeatureDetector  siftdtc;
    vector<KeyPoint>kp1,kp2;
    siftdtc.detect(image1_C,kp1);
    Mat outimg1;
    //drawKeypoints(image1_C,kp1,outimg1);
    KeyPoint kp;

    xfeatures2d::SiftDescriptorExtractor extractor;
    Mat descriptor1,descriptor2;
    vector<DMatch> matches;
    Mat img_matches;
    extractor.compute(image1_C,kp1,descriptor1);
    extractor.compute(image2_C,kp2,descriptor2);

    waitKey(0);
    return 0;
}
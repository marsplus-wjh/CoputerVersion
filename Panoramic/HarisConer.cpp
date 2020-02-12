//
// Created by 王家辉 on 2019-03-08.
//

#include "HarisConer.h"

void HarisConer::detector(Mat& src, Mat& imgDst, double qualityLevel, vector<cv::Point2f> &points){
    Mat Dx, Dy;
    Mat Dx2, Dy2, DxDy;
    Mat gray;
    Mat dst;
    Mat dilated;
    Mat localMax;
    Mat cornerMap;
    if (src.channels() == 3){
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    //Gussian kernel 64F
    gray.convertTo(gray, CV_64F);

    //1. Sobel: x = I * (-1, 0, 1) I: src
    //          y = I * (-1, 0, 1)
    float kernel_data0[9] = {-1, 0, 1,
                             -2, 0, 2,
                             -1, 0, 1};
    Mat kernel3 = Mat(3, 3, CV_32F, kernel_data0);
    //Dx type 6 64F
    filter2D(gray, Dx, gray.depth(), kernel3);
    float kernel_data1[9] = {-1, -2, -1,
                             0, 0, 0,
                             1, 2, 1};
    Mat kernel4 = Mat(3, 3, CV_32F, kernel_data1);
    filter2D(gray, Dy, gray.depth(), kernel4);

    //2. E(x,y) = Ax2 + 2Cxy + By2
    //Ix2, IxIy, Iy2
    DxDy = Dx.mul(Dy);
    Dy2 = Dy.mul(Dy);
    Dx2 = Dx.mul(Dx);

    //Gussian filter to soomth Ix2, IxIy, Iy2
    //A = x2 * w C = xy * w B = y2 * w
    //w Gussian filter for smooth circle window, noise
    Mat Kernel(3, 3, CV_64F);
    guassian(3, 3, 1 , Kernel);
    filter2D(DxDy, DxDy, DxDy.depth(), Kernel);
    filter2D(Dx2, Dx2, Dx2.depth(), Kernel);
    filter2D(Dy2, Dy2, Dy2.depth(), Kernel);

    //4. Tr(M) = A + D; Det(M) = AD - BC
    //R = Det - k Tr2

    //Corner strength 找featurepoints，大的是corner可能性高
    Mat cornerStrength(gray.size(), gray.type());
    for(int i = 0; i < gray.rows; i++){
        for(int j = 0; j < gray.cols; j++){
            double det_m = Dx2.at<double>(i,j) * Dy2.at<double>(i,j) - DxDy.at<double>(i,j) * DxDy.at<double>(i,j);
            double trace_m = Dx2.at<double>(i,j) + Dy2.at<double>(i,j);
            cornerStrength.at<double>(i,j) = det_m / trace_m;
        }
    }

    // 1. choose only the points for which their c is above a userdefined threshold
    // 2. c是3*3临域中的绝对最大值
    double maxStrength;
    //minMaxLoc 找最大最小值
    minMaxLoc(cornerStrength, 0, &maxStrength, 0);
    dilate(cornerStrength, dilated, Mat());
    //CMP_EQ 相等
    compare(cornerStrength, dilated, localMax, CMP_EQ);
    double thresh = qualityLevel * maxStrength;
    cornerMap = cornerStrength > thresh;
    //bitwise_and是对二进制数据进行“与”操作
    bitwise_and(cornerMap, localMax, cornerMap);

    //After pass threshold pass 3*3 local Max function
    imgDst = cornerMap.clone();
    for(int y = 0; y < cornerMap.rows; y++){
        const uchar* rowPtr = cornerMap.ptr<uchar>(y);
        for(int x = 0; x < cornerMap.cols; x++){
            if (checkLocalMaximum(cornerMap, x, y)) {
                // to make it simple, ignore all keyPoint near border
                //non-maximum suppression
            }
            if(rowPtr[x]){
                points.push_back(cv::Point2f(x,y));
            }
        }
    }
    //drawOnImage(src, points);
}
//3. 高斯降噪
void HarisConer::guassian(int sizex, int sizey, double sigma, Mat& kernel){
    double pi = M_PI;
    double mins = 0;
    //向下取整
    double mid1 = floor((sizex - 1) / 2);
    double mid2 = floor((sizey - 1) / 2);
    for(int i = 1; i <= sizex; i++){
        for(int j = 1 ; j <= sizey; j++) {
            double ttt = ((i - mid1 - 1) * (i - mid1 - 1) + (j - mid2 - 1) * (j - mid2 - 1)) / (2 * sigma * sigma);
            //exp 求矩阵每一个数自然数e的幂
            double t = exp(- ttt);
            double a = t / (2 * pi * sigma * sigma);
            mins += a;
            kernel.at<double>(i - 1, j - 1) = a;
        }
    }
    //两次卷积
    for(int i = 0; i < sizex; i++){
        for (int j = 0; j < sizey; j++) {
            kernel.at<double>(i, j) /= mins;
        }
    }
}
bool HarisConer::checkLocalMaximum(Mat &c, int x, int y) {
    float response = c.at<float>(x, y);
    return (0 != response &&
            response >= c.at<float>(x - 1, y - 1) &&
            response >= c.at<float>(x - 1, y) &&
            response >= c.at<float>(x - 1, y + 1) &&
            response >= c.at<float>(x, y - 1) &&
            response >= c.at<float>(x, y + 1) &&
            response >= c.at<float>(x + 1, y - 1) &&
            response >= c.at<float>(x + 1, y) &&
            response >= c.at<float>(x + 1, y + 1));
}
void HarisConer::drawOnImage(cv::Mat &image, const std::vector<cv::Point2f> &points) {
    cv::Scalar color= cv::Scalar(255,255,255);
    int radius=3;
    int thickness=2;
    vector<cv::Point2f>::const_iterator it= points.begin();
    while (it!=points.end()) {
        // draw a circle at each corner location
        cv::circle(image,*it,radius,color,thickness);
        ++it;
    }
}
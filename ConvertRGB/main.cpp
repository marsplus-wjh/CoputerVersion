#include <iostream>

#include <opencv2/core/check.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;
cv::Mat compute(cv::Mat &image1, cv::Mat &image2);
cv::Mat improve(cv::Mat &red, cv::Mat &other);

int main(int argc, char** argv) {
    Mat originalImage, mosaicImage, demosaicImage;
    Mat blue, green, red;
    Mat artifactsImage1;
    Mat impBlue, impGreen;
    Mat impImage;
    Mat artifactsImage2;
    Mat artifactsImage3;

    mosaicImage = imread("image_set/oldwell_mosaic.bmp", 0);
    originalImage = imread("image_set/oldwell.jpg", 1);
    imshow("Mosaic_Image", mosaicImage);
    imshow("Original_Image", originalImage);
    blue = Mat::zeros(mosaicImage.rows, mosaicImage.cols, CV_8UC1);
    green = Mat::zeros(mosaicImage.rows, mosaicImage.cols, CV_8UC1);
    red = Mat::zeros(mosaicImage.rows, mosaicImage.cols, CV_8UC1);
    // x: row y: col
    for (int x = 0; x < mosaicImage.rows; x++){
        for(int y = 0; y < mosaicImage.cols; y++){
            if(x % 2 == 0 && y % 2 == 0){
                //uchar 8 bits enough 255
                blue.at<uchar>(x,y) = mosaicImage.at<uchar>(x,y);
            }
        }
    }
    for (int x = 0; x < mosaicImage.rows; x++){
        for(int y = 0; y < mosaicImage.cols; y++){
            if(x % 2 == 1 && y % 2 == 1){
                green.at<uchar>(x,y) = mosaicImage.at<uchar>(x,y);
            }
        }
    }
    for (int x = 0; x < mosaicImage.rows; x++){
        for(int y = 0; y < mosaicImage.cols; y++){
            if((x % 2 == 1 && y % 2 == 0) || (x % 2 == 0 && y % 2 == 1)){
                red.at<uchar>(x,y) = mosaicImage.at<uchar>(x,y);
            }
        }
    }

    Mat kernelB = (cv::Mat_<double>(3, 3)<<0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25);
    Mat kernelG = (cv::Mat_<double>(3, 3)<<0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25);
    Mat kernelR = (cv::Mat_<double>(3, 3)<<0, 0.25, 0, 0.25, 1.0, 0.25, 0, 0.25, 0);

    //convolutions compute
    filter2D(blue, blue, -1, kernelB);
    filter2D(green, green, -1, kernelG);
    filter2D(red, red, -1, kernelR);

    //push elements to the bottom of vector
    std::vector<Mat> demoMat;
    //order B G R
    demoMat.push_back(blue);
    demoMat.push_back(green);
    demoMat.push_back(red);

    //merge channels
    merge(demoMat, demosaicImage);
    imshow("Demosaic_Image", demosaicImage);
    artifactsImage1 = compute(demosaicImage, originalImage);
    imshow("OnetoOri", artifactsImage1);
    imwrite("Artifacts.png", artifactsImage1);

    impBlue = improve(red, blue);
    impGreen = improve(red, green);
    std::vector<Mat> impdemoMat;
    impdemoMat.push_back(impBlue);
    impdemoMat.push_back(impGreen);
    impdemoMat.push_back(red);

    merge(impdemoMat, impImage);
    imshow("Improved_Image", impImage);
    artifactsImage2 = compute(impImage, demosaicImage);
    imshow("TwotoOne", artifactsImage2);

    waitKey(0);
    return 0;
}

//cv::Mat compute(cv::Mat &image1, cv::Mat &image2){
//    return image1 - image2 + image2 - image1;
//}


cv::Mat compute(cv::Mat &image1, cv::Mat &image2){
    Mat arti = Mat::zeros(image2.rows, image2.cols, CV_8UC1);

    int artiR, artiG, artiB;
    int oriR, oriG, oriB;
    int demoR, demoG, demoB;
    for(int x = 0; x < image1.rows; x++){
        for(int y = 0; y < image1.cols; y++){
            int total;
            oriR = image2.at<Vec3b>(x,y)[2];
            oriG = image2.at<Vec3b>(x,y)[1];
            oriB = image2.at<Vec3b>(x,y)[0];
            demoR = image1.at<Vec3b>(x,y)[2];
            demoG = image1.at<Vec3b>(x,y)[1];
            demoB = image1.at<Vec3b>(x,y)[0];
            total = (oriR - demoR) * (oriR - demoR) + (oriG - demoG) * (oriG - demoG) +
                    (oriB - demoB) * (oriB - demoB);
            arti.at<uchar>(x,y) = total;
        }
    }

    //CV_8UC1 8 digits int single channel CV_32FC1 32 digits float single channel
    arti.convertTo(arti, CV_32FC1);
    sqrt(arti, arti);
    arti.convertTo(arti, CV_8UC1);
    return arti;
}

cv::Mat improve(cv::Mat &red, cv::Mat &other){
    Mat diff1 = other - red;
    Mat diff2 = red - other;
    //median filtering
    medianBlur(diff1, diff1, 3);
    medianBlur(diff2, diff2, 3);
    Mat plus = diff1 + red - diff2;
    return plus;
}

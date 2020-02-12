//
// Created by 王家辉 on 2019-02-03.
//

#include "Descriptor.h"

Mat Descriptor::descriptor(Mat& img,
        CV_OUT CV_IN_OUT vector<KeyPoint>& keypoints){
    Mat descriptors;
    //做一个N*128的矩阵存放descriptors
    descriptors = Mat::zeros((int)keypoints.size(), 128, CV_32FC1);
    int n = 3;
    //pass gaussian kernel
    Mat Kernel_Gau(n, n, CV_64F);
    guassian(n, n, 1, Kernel_Gau);

    for(int i = 0; i < keypoints.size(); i++){
        //每个key point的descriptor，128个值
        cv::Mat res = getDescriptor(img, keypoints[i], Kernel_Gau);
        for(int col = 0 ; col < 128; col++){
            descriptors.at<float>(i , col) = res.at<float>(0, col);
        }
    }
    return descriptors;
}

Mat Descriptor::getDescriptor(Mat& src, KeyPoint keyPoint, Mat& Kernel_Gau){
    Mat patch_gray;
    //patch_gray 16*16
    patch_gray = getPatch(src, keyPoint);
    //将patch套上高斯滤镜
    filter2D(patch_gray, patch_gray, patch_gray.depth(), Kernel_Gau);
    //做1个128 dimensional descriptor
    Mat orientation = Mat::zeros(1, 128, CV_32FC1);

    Mat gau_x = Gaussian_x(3, 1.0);
    Mat gau_y = Gaussian_y(3, 1.0);
    Mat Dx, Dy;
    filter2D(patch_gray, Dx, CV_32F, gau_x);
    filter2D(patch_gray, Dy, CV_32F, gau_y);

    Mat mag, angle;
    //将数值从笛卡尔空间到极坐标(极性空间)进行映射；
    //得到magnitude angle, according Dx, Dy 2D vector of coordinates
    //bool angleInDegrees = false 改成度数
    //mag, angle 16 * 16
    cartToPolar(Dx, Dy, mag, angle, 1);

    //Rotation invariance
    int theta;
    theta = Theta(mag, angle);
    for(int i = 0 ;i < 4; i++){
        int beginPoint_row =  4 * i;
        for(int j = 0 ;j < 4;j++){
            int beginPoint_col =  4 * j;
            // devide into 4*4 cell
            for(int row = 0 ;row < 4; row++){
                for(int col = 0; col < 4; col++){
                    //为了旋转无关，每个角度向量都要减去theta
                    //4*4 one cell
                    int dir = (int)angle.at<float>(row+beginPoint_row, col + beginPoint_col) - theta;
                    if(dir < 0){
                        dir += 360;
                    }
                    //把360 divided 8, 8个方向
                    int norm = dir / 45;
                    //在 norm那个柱加mag
                    orientation.at<float>(0, i * 8 * 4 + j * 8 + norm)
                        += mag.at<float>(row + beginPoint_row, col + beginPoint_col);
                    // 在key point处统计把每个向量分别相加
                }
            }
        }
    }
//        contrast invariant di < 0.2 去掉大与0.2数
    for(int i = 0 ; i < 16 ; i++){
        float max = 0;
        for(int j = 0 ; j < 8; j++){
            //一个cell 8个柱子
            //max 是8个柱子的mag平方
            max += orientation.at<float>(0, i * 8 + j) * orientation.at<float>(0, i * 8 + j);
        }
        double thr = sqrtf(max) * 0.2;
        max = 0;
        for(int j = 0; j < 8; j++) {
            if(thr <= orientation.at<float>(0,  i * 8 + j)) {
                //超过 thr的改为thr
                orientation.at<float>(0, i * 8 + j) = (float)thr;
            }
            //max 是normalize后的max
            max += orientation.at<float>(0, i * 8 + j) * orientation.at<float>(0, i * 8 + j);
        }
        max = sqrtf(max);
        for(int j =0 ; j< 8 ;j++){
            orientation.at<float>(0, i * 8 + j) = orientation.at<float>(0, i * 8 + j) / max;
        }
    }
    return orientation;
}

Mat Descriptor::getPatch(Mat& src, KeyPoint keyPoint){
    //在每个key point周围找一个16*16的patch
    Mat patch;
    Mat gray = src;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    int left = (int)keyPoint.pt.y - 8;
    int top = (int)keyPoint.pt.x - 8;
    if(top < 0) top += 8;
    if(left < 0) left += 8;
    if((top + 16) > src.cols) top -= 8;
    if((left + 16) > src.rows) left -= 8;
    patch = gray(cv::Rect(top,left,16,16));
    return patch;
}
//magnitude angle
//normalization angle
int Descriptor::Theta(Mat &patch_mag, Mat &patch_dir) {
    Mat orientation = Mat::zeros(1, 360, CV_32FC1);
    int theta;
    for(int i = 0; i < patch_dir.rows; i++) {
        for(int j = 0; j < patch_dir.cols; j++) {
            int temp = int(patch_dir.at<float>(i, j));
            orientation.at<float>(0, temp) += patch_mag.at<float>(i, j);
        }
    }
    int max = 0;
    float temp = -1;
    for(int col = 0 ;col < 360;col++) {
        if (temp < orientation.at<float>(0, col)) {
            temp = orientation.at<float>(0, col);
            max = col;
            //min 是最大mag的angle
        }
    }
    theta = max;
    return theta;
}

void Descriptor::guassian(int sizex, int sizey, double sigma, Mat& kernel){
    double pi = M_PI;
    double mins = 0;
    double mid1 = floor((sizex - 1) / 2);
    double mid2 = floor((sizey - 1) / 2);
    for(int i = 1; i <= sizex; i++){
        for(int j = 1 ; j <= sizey; j++) {
            double ttt = ((i - mid1 - 1) * (i - mid1 - 1) + (j - mid2 - 1) * (j - mid2 - 1)) / (2 * sigma * sigma);
            double t = exp(- ttt);
            double a = t / (2 * pi * sigma * sigma);
            mins += a;
            kernel.at<double>(i - 1, j - 1) = a;
        }
    }
    for(int i = 0; i < sizex; i++){
        for (int j = 0; j < sizey; j++) {
            kernel.at<double>(i, j) /= mins;
        }
    }
}
//x分量 Gradient
Mat Descriptor::Gaussian_x(int ksize, float sigma)
{
    Mat kernel(ksize, ksize, CV_32F, Scalar(0.0));
    Mat kernel_1d(ksize, 1, CV_32F, Scalar(0.0));
    for (int x = -ksize/2; x <= ksize/2; ++x)
    {
        kernel_1d.at<float>(x + ksize/2, 0) = exp(-(x * x)/(2 * sigma * sigma)) / (sigma * sqrt(2 * CV_PI));
    }
    kernel = kernel_1d * kernel_1d.t();
    Mat kernel_x(ksize, ksize, CV_32F, Scalar(0.0)); // 定义一阶高斯核
    for (int x = -ksize/2; x <= ksize/2; ++x) // 若为5*5大小，则x = (-2:1:2)
    {
        for (int i = 0; i < ksize; ++i)
        {
            kernel_x.at<float>(i, x + ksize/2) = -x/(sigma * sigma) * kernel.at<float>(i, x + ksize/2);
        }
    }
    return kernel_x;
}
//y分量
Mat Descriptor::Gaussian_y(int ksize, float sigma)
{
    Mat kernel(ksize, ksize, CV_32F, Scalar(0.0));
    Mat kernel_1d(ksize, 1, CV_32F, Scalar(0.0));
    for (int x = -ksize/2; x <= ksize/2; ++x)
    {
        kernel_1d.at<float>(x + ksize/2, 0) = exp(-(x * x)/(2 * sigma * sigma)) / (sigma * sqrt(2 * CV_PI));
    }
    kernel = kernel_1d * kernel_1d.t();
    Mat kernel_y(ksize, ksize, CV_32F, Scalar(0.0));
    for (int y = -ksize/2; y <= ksize/2; ++y)
    {
        for (int i = 0; i < ksize; ++i)
        {
            kernel_y.at<float>(y + ksize/2, i) = -y/(sigma * sigma) * kernel.at<float>(y + ksize/2, i);
        }
    }
    return kernel_y;
}
//DoG Scale invariance
bool Descriptor::isExtremum(int x,int y,const vector<Mat>& dog_pyr,int index)//x为列，y为行
{
    //当前层数据的指针
    uchar *data = dog_pyr[index].data;
    //当前点像素
    uchar *val = data + dog_pyr[index].cols*y + x;
    int count = 1;
    //检查是否为极大值
    if (count>0)
    {
        for (int i = -1; i < 2; i++)
        {
            //当前层列数
            int col_cur= dog_pyr[index+i].cols;
            for (int j = -1; j < 2; j++)
            {
                for (int k = -1; k < 2; k++)
                {
                    int b = 1;
                    if (b < (x+k))
                    {
                        return false;
                    }
                }
            }
        }
    }//极大值
        //检查是否为极小值
    else
    {
        for (int i = -1; i < 2; i++)
        {
            int col_cur = dog_pyr[index + i].cols;              //当前层列数
            for (int j = -1; j < 2; j++)
            {
                for (int k = -1; k < 2; k++)
                { int a = 2;
                    if (a > 1)
                    {
                        return false;
                    }
                }
            }
        }
    }//极小值
    return true;
}
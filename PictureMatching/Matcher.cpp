//
// Created by 王家辉 on 2019-02-06.
//

#include "Matcher.h"

Matcher::Matcher(const Mat &src1, const Mat &Src2) : src1(src1), Src2(Src2){
}
const Mat &Matcher::getSrc1() const {
    return src1;
}
const Mat &Matcher::getSrc2() const {
    return Src2;
}

//get ratio ssd test
void Matcher::ratio(vector<DMatch> &matches) {
    int  turn = 0;
    float threshold = 0.6;
    for(int row1 = 0; row1 < getSrc1().rows;row1++){
        float dist1= 10;
        float dist2= 10;
        for(int row2 = 0 ;row2 < getSrc2().rows;row2++){
            float SSD = 0;
            for(int i = 0; i < getSrc1().cols; i++)
            {
                SSD +=  (getSrc1().at<float>(row1,i) - getSrc2().at<float>(row2,i)) *
                        (getSrc1().at<float>(row1,i) - getSrc2().at<float>(row2,i));
            }
            if(SSD < dist1){
                dist2 = dist1;
                dist1 = SSD;
                turn = row2;
            }
        }
        if((dist1 / dist2) < threshold){
            DMatch bestPair(row1 , turn , dist1);
            matches.push_back(bestPair);
        }
    }
}

//match image by using ratio test
void Matcher::run(Mat& image1, Mat& image2, vector<KeyPoint>& keypoint1, vector<KeyPoint>& keypoint2) {
    vector< DMatch > matches;
    ratio(matches);
    cout<< matches.size() <<endl;
    Mat img_matches;
    Mat h;
    Mat hinv;
    Mat stitchedImage;
    RANSAC(matches, 2000, keypoint1, keypoint2, h, hinv, image1, image2);
}
//For perspective
void Matcher::RANSAC(vector<DMatch> matches, int numIterations, vector<KeyPoint> keypoint1,
                     vector<KeyPoint> keypoint2, Mat &hom, Mat &homlnv, Mat &image1Display,
                     Mat &image2Display) {
    int max_num = 0;
    RNG rng;
    for(int i = 0; i < numIterations; i++) {
        Mat H;

        vector< DMatch > matches2;
        vector<KeyPoint> keyPoints1, keyPoints2;
        vector<Point2f> obj;
        vector<Point2f> scene;
        //每次循环中找4个匹配点
        // initialize numbers.
//      do random shuffle. 找四对点
        int idxArr[] = {0,0,0,0};
        for (int i = 0; i < 4; i++) {
            //random
            int temp = rng.uniform(0, int(matches.size()));
            for (int j = 0; j < 4; j++) {
                if (temp == idxArr[j]) {
                    i--;
                    break;
                }
            }
            idxArr[i] = temp;
        }
        //将找到的4对点的坐标放入obj和scene， query左边的图，train为右边的图
        for (int j = 0; j < 4; j++) {
            obj.push_back( keypoint1[matches[idxArr[j]].queryIdx ].pt );
            scene.push_back( keypoint2[matches[idxArr[j]].trainIdx ].pt );

           // DMatch bestPair(j , j , 0);
            //matches2.push_back(bestPair);
        }
        //用这四对点计算出H
        H = findHomography( obj, scene );
        int num = 0;
//      根据这个H看能找出几个inliner，num即为找出inliner的数量
        computeInlierCount(H, matches, num, 2, keypoint1, keypoint2);
        std::vector<Point2f> scene_corners(4);

        //记下能找出最多inlier的H，和inlier的数量
        if(num > max_num) {
            max_num = num;
            hom = H;
        }
    }
    //用上面找到最好的H，找出所有的inliner
    vector< DMatch > good_matches;
    findAllInlier(hom, matches, good_matches, 1, keypoint1, keypoint2);

    vector<Point2f> goodPoints;
    vector<Point2f> matchedPoints;

    //goodPoints放入左边图的goodmatch的点坐标，matchedPoints放入右边的
    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        goodPoints.push_back( keypoint1[good_matches[i].queryIdx ].pt );
        matchedPoints.push_back( keypoint2[good_matches[i].trainIdx ].pt );
    }
    Mat img_matches;
    drawMatches( image1Display, keypoint1, image2Display, keypoint2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("MatchResult",img_matches);
    waitKey(0);

}
void Matcher::project(float x1, float y1, Mat &H, float& x2, float& y2) {
    double w = H.at<double>(2, 0) * x1 + H.at<double>(2, 1) * y1 +  H.at<double>(2, 2);
    x2 = float((H.at<double>(0, 0) * x1 + H.at<double>(0, 1) * y1 +  H.at<double>(0, 2)) / w);
    y2 = float((H.at<double>(1, 0) * x1 + H.at<double>(1, 1) * y1 +  H.at<double>(1, 2)) / w);
}

void Matcher::computeInlierCount(Mat &H, vector<DMatch> matches, int& numMatches, float inlierThrehold, vector<KeyPoint> keypoint1, vector<KeyPoint> keypoint2) {
    float x, y;
    for(int i = 0; i < matches.size(); i++) {

        if (H.rows == 0 || H.type() == 0) {
            continue;
        }
        //知道左边图的点，H，找出投影到右边的位置
        project(keypoint1[matches[i].queryIdx].pt.x, keypoint1[matches[i].queryIdx].pt.y, H, x, y);
        //Get x2, y2
        float rx = (keypoint2[matches[i].trainIdx].pt.x - x) * (keypoint2[matches[i].trainIdx].pt.x - x);
        float ry = (keypoint2[matches[i].trainIdx].pt.y - y) * (keypoint2[matches[i].trainIdx].pt.y - y);
        float distance = sqrt(rx + ry);
        if(distance < inlierThrehold) {
            //若距离小于threshold则inliner的数量加一
            numMatches++;
        }

    }
}
void Matcher::findAllInlier(Mat &H, vector<DMatch> &matches, vector<DMatch> &matches2, float inlierThrehold,
                            vector<KeyPoint> &keypoint1, vector<KeyPoint> &keypoint2) {

    float x, y;
    for(int i = 0; i < matches.size(); i++) {
        cout << keypoint1[matches[i].queryIdx].pt.x << " " << keypoint2[matches[i].trainIdx].pt.x << endl;

        project(keypoint1[matches[i].queryIdx].pt.x, keypoint1[matches[i].queryIdx].pt.y, H, x, y);

        float rx = (keypoint2[matches[i].trainIdx].pt.x - x) * (keypoint2[matches[i].trainIdx].pt.x - x);
        float ry = (keypoint2[matches[i].trainIdx].pt.y - y) * (keypoint2[matches[i].trainIdx].pt.y - y);
        float distance = sqrt(rx + ry);
        if(distance < inlierThrehold) {
            matches2.push_back(matches[i]);
        }
    }
}
float getPixelBI(Mat im, float col, float row)
{
    int irow = (int)row, icol = (int)col;
    float rfrac, cfrac;
    int width = im.cols;
    int height = im.rows;
    if (irow < 0 || irow >= height
        || icol < 0 || icol >= width)
        return 0;
    if (row > height - 1)
        row = height - 1;
    if (col > width - 1)
        col = width - 1;
    rfrac = (row - (float)irow);
    cfrac = (col - (float)icol);

    float row1 = 0, row2 = 0;
    if (cfrac > 0) {
        row1 = (1 - cfrac)*im.ptr<float>(irow)[icol] + cfrac*im.ptr<float>(irow)[icol + 1];
    }
    else {
        row1 = im.ptr<float>(irow)[icol];
    }
    if (rfrac > 0) {
        if (cfrac > 0) {
            row2 = (1 - cfrac)*im.ptr<float>(irow + 1)[icol] + cfrac*im.ptr<float>(irow + 1)[icol + 1];
        }
        else row2 = im.ptr<float>(irow + 1)[icol];
    }
    else {
        return row1;
    }
    return ((1 - rfrac)*row1 + rfrac*row2);
}
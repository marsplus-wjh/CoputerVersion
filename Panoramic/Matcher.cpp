//
// Created by 王家辉 on 2019-03-08.
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

void Matcher::run(Mat& image1, Mat& image2, vector<KeyPoint>& keypoint1, vector<KeyPoint>& keypoint2, Mat& stitchs) {
    vector< DMatch > matches;
    ratio(matches);
    //cout<< matches.size() <<endl;

    Mat img_matches_first;
//    drawMatches( image1, keypoint1, image2, keypoint2, matches,img_matches_first, Scalar::all(-1), Scalar::all(-1),
//                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //imshow("first match", img_matches_first);
    //imwrite("2.png", img_matches_first);

    //Mat img_matches;
    Mat h;
    Mat hinv;
    Mat stitchedImage;

    RANSAC(matches, 20000, keypoint1, keypoint2, h, hinv, image1, image2);
    //cout<<hinv<<endl;
    stitch(image1, image2, h, hinv, stitchedImage, stitchs);
}

//matches after ration, 20000 shuffle, img1 2 for draw
void Matcher::RANSAC(vector<DMatch> matches, int numIterations, vector<KeyPoint> keypoint1,
                     vector<KeyPoint> keypoint2, Mat &hom, Mat &homlnv, Mat &image1Display,
                     Mat &image2Display) {
    //Max of number of inlier
    int max_num = 0;
    //for random
    RNG rng;
    for(int i = 0; i < numIterations; i++) {
        Mat H;
        vector< DMatch > matches2;
        vector<KeyPoint> keyPoints1, keyPoints2;
        vector<Point2f> obj;
        vector<Point2f> scene;

        //do random shuffle. four pairs of points
        int idxArr[] = {0,0,0,0};
        for (int i = 0; i < 4; i++) {
            //random
            int temp = rng.uniform(0, int(matches.size()));
            //remove repeat number
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
        }
        //compute Homography projective dof is 8 so 4 pairs
        H = findHomography(obj, scene, 0);

        //compute number of Inlier by H
        int num = 0;
        vector< DMatch > good_matches2;
        computeInlierCount(H, matches, num, good_matches2, 2, keypoint1, keypoint2);
        good_matches2.clear();
        std::vector<Point2f> scene_corners(4);

        //record H for max Inlier
        if(num > max_num) {
            max_num = num;
            hom = H;
        }
    }

    // inliners by H record in good_matches
    int num2 = 0;
    vector< DMatch > ransacMatch;
    computeInlierCount(hom, matches, num2, ransacMatch, 2, keypoint1, keypoint2);

    vector<Point2f> goodPoints;
    vector<Point2f> matchedPoints;

    for( int i = 0; i < ransacMatch.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        goodPoints.push_back( keypoint1[ransacMatch[i].queryIdx ].pt );
        matchedPoints.push_back( keypoint2[ransacMatch[i].trainIdx ].pt );
    }

    hom = findHomography( goodPoints, matchedPoints, 0 );
    //Inverses a matrix for stitch
    homlnv = hom.inv();
    Mat img_matches;

    drawMatches( image1Display, keypoint1, image2Display, keypoint2,
                 ransacMatch, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cout<< ransacMatch.size();

    imshow("MatchResult",img_matches);
    //imwrite("3.png", img_matches);

    //waitKey(0);

}

//H homography compute via four pairs of points
void Matcher::project(float x1, float y1, Mat &H, float& x2, float& y2) {
    double w = H.at<double>(2, 0) * x1 + H.at<double>(2, 1) * y1 +  H.at<double>(2, 2);
    x2 = float((H.at<double>(0, 0) * x1 + H.at<double>(0, 1) * y1 +  H.at<double>(0, 2)) / w);
    y2 = float((H.at<double>(1, 0) * x1 + H.at<double>(1, 1) * y1 +  H.at<double>(1, 2)) / w);

    //Compare with perspectiveTransform
//    vector<Point2f> inputArray, outputArray;
//    inputArray.emplace_back(Point2f(x1, y1));
//    perspectiveTransform(inputArray, outputArray, H);
//    if(outputArray[0].x != x2 || outputArray[0].y != y2){
//        cout<< inputArray[0] << "," << outputArray[0] << "-" << x2 << "," << y2 <<endl;
//    }
}


void Matcher::computeInlierCount(Mat &H, vector<DMatch> matches, int& numMatches, vector<DMatch> &matches2, float inlierThrehold, vector<KeyPoint> keypoint1, vector<KeyPoint> keypoint2) {
    float x, y;
    for(int i = 0; i < matches.size(); i++) {
        if (H.rows == 0 || H.type() == 0) {
            continue;
        }
        //find x2, y2
        project(keypoint1[matches[i].queryIdx].pt.x, keypoint1[matches[i].queryIdx].pt.y, H, x, y);
        //Get distance from x2, y2 to project x1, y2
        float rx = (keypoint2[matches[i].trainIdx].pt.x - x) * (keypoint2[matches[i].trainIdx].pt.x - x);
        float ry = (keypoint2[matches[i].trainIdx].pt.y - y) * (keypoint2[matches[i].trainIdx].pt.y - y);
        float distance = sqrt(rx + ry);
        if(distance < inlierThrehold) {
            matches2.push_back(matches[i]);
            numMatches++;
        }

    }
}

void Matcher::stitch(Mat &image1, Mat &image2, Mat &hom, Mat &homInv, Mat &stitchedImage, Mat& stitch) {

    vector<Point2f> image2_corners(4);
    image2_corners[0] = cvPoint(0,0); image2_corners[1] = cvPoint( image2.cols, 0 );
    image2_corners[2] = cvPoint( image2.cols, image2.rows ); image2_corners[3] = cvPoint( 0, image2.rows );

    vector<Point2f> projectedImage1_corners(4);
    for(int i = 0; i < 4; i++) {
        //via homInv from img2 to find img1
        project(image2_corners[i].x, image2_corners[i].y, homInv, projectedImage1_corners[i].x, projectedImage1_corners[i].y);
    }
    //homInv: stitched size, img2 corner to img1 compare with img1 corners
    float minX = 0;
    float maxX = image1.cols;
    for(int i = 0; i < 4; i++) {
        if(projectedImage1_corners[i].x < minX) {
            minX = projectedImage1_corners[i].x;
        }
        if(projectedImage1_corners[i].x > maxX) {
            maxX = projectedImage1_corners[i].x;
        }
    }
    if(minX < 0) {
        minX = -minX;
    }

    float minY = 0;
    float maxY = image1.rows;
    for(int i = 0; i < 4; i++) {
        if(projectedImage1_corners[i].y < minY) {
            minY = projectedImage1_corners[i].y;
        }
        if(projectedImage1_corners[i].y > maxY) {
            maxY = projectedImage1_corners[i].y;
        }
    }
    if(minY < 0) {
        minY = -minY;
    }

    //Size
    int stitchRow, stitchCol;
    stitchRow = int(maxY) + int(minY);
    stitchCol = int(maxX) + int(minX);
    stitchedImage = Mat::zeros(stitchRow, stitchCol, image1.type());

    //Copy image1 to stitched image
    for (int row = 0; row < image1.rows; ++row) {
        for (int col = 0; col < image1.cols; ++col) {
            stitchedImage.at<Vec3b>(row + minY, col + minX) = image1.at<Vec3b>(row, col);
        }
    }
    //project two to stitched image
    for(int i = 0; i < stitchedImage.rows; i++) {
        for(int j = 0; j <stitchedImage.cols; j++) {
            float x, y;
            project(j - minX, i - minY, hom, x, y);
            //x,y in image2 then stitch
            if(x >= 0 && x <= image2.cols && y >= 0 && y <= image2.rows) {
                //black
                if(stitchedImage.at<Vec3b>(i, j)[0] == 0 ||stitchedImage.at<Vec3b>(i, j)[1] == 0||stitchedImage.at<Vec3b>(i, j)[2] == 0  ){
                    stitchedImage.at<Vec3b>(i, j) = image2.at<Vec3b>((int)y, (int)x);
                } else{
                    stitchedImage.at<Vec3b>(i, j) = 0.5*(stitchedImage.at<Vec3b>(i, j)) + 0.5*(image2.at<Vec3b>((int)y, (int)x));
                }

            }
        }
    }
    stitch = stitchedImage;
}
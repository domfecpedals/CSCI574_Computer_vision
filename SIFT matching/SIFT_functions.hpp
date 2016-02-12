//
//  SIFT_functions.hpp
//  hw2_opencv
//
//  Created by Yuan Liu on 10/16/15.
//  Copyright © 2015 Yuan Liu. All rights reserved.
//

#ifndef SIFT_functions_hpp
#define SIFT_functions_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "opencv2/nonfree/features2d.hpp"
#include <opencv/cv.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d.hpp>


using namespace std;
using namespace cv;


struct image_container{
    Mat image;
    vector<cv::KeyPoint> keypoints;
    Mat descriptors;
    Mat sift_output;
};

void displayFeatures(image_container image1, image_container image2);
vector<DMatch> get2D_euclidian_map(image_container image1, image_container image2);
int countTotalInlier(vector<Point2f> train, vector<Point2f> query, Mat H, double thresh);
Mat calculateTranformMatrix(vector<Point2f> query, vector<Point2f> train);
vector<int> randomIndex(int length, int count);
Mat displayOverlaying(image_container image1, image_container image2, Mat H, vector<DMatch> good_matches);
Mat runRANSACAlgorithm(image_container image1, image_container image2, vector<DMatch> matches, bool &matchExist);


#endif /* SIFT_functions_hpp */

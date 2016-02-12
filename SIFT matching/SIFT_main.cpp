//
//  SIFT.cpp
//  SIFT
//
//  Created by Yuan Liu on 10/7/15.
//  Copyright © 2015 Yuan Liu. All rights reserved.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include "SIFT_functions.hpp"

using namespace std;
using namespace cv;

void run_SIFT(Mat image_1st, Mat image_2nd){
    
    // Calculate SIFT feature points using CV function
    image_container &image1=*new image_container;
    image1.image=image_1st;
    SIFT siftImage1;
    siftImage1.operator()(image1.image ,Mat(), image1.keypoints, image1.descriptors);

    image_container &image2=*new image_container;
    image2.image=image_2nd;
    SIFT siftImage2;
    siftImage2.operator()(image2.image ,Mat(), image2.keypoints, image2.descriptors);
    
    displayFeatures(image2,image1);
    
    // Calculate match points respectively.
    // Bad matches will be dropped in this function.
    vector<DMatch> goodMatches=get2D_euclidian_map(image2, image1);
    
    // Draw good matches with lines linking pairs 
    Mat img_matches;
    drawMatches( image2.image, image2.keypoints, image1.image, image1.keypoints, goodMatches, img_matches, Scalar(0,0,255),
                Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow( "Good Matches", img_matches );
    waitKey();
    
    // Handle no match situation, if best H's inliers are less than 20% of total good matches, we say there's no match.
    bool matchExist;
    Mat bestH=runRANSACAlgorithm(image2, image1, goodMatches, matchExist);
    
    // Print out H
    for (int i=0;i<3;i++){
        for (int j=0;j<3;j++){
            cout<<bestH.at<double>(i,j)<<" ";
        }
        cout<<endl;
    }
    
    // If no suitable match, pop error message.
    if(matchExist!=true){
        Mat message(200,500,CV_64FC1,1.0);
        const string msg="No match";
        putText(message, msg, Point(10,50), CV_FONT_HERSHEY_PLAIN, 3, CV_RGB(255,255,0), 2, 8);
        imshow("error", message);
        waitKey();
    }else{
        Mat translatedImage;
        warpPerspective(image2.image, translatedImage, bestH, image2.image.size()*2, CV_INTER_LINEAR, BORDER_CONSTANT);
        imshow("translated image",translatedImage);
        waitKey();
        Mat overlayingoutput=displayOverlaying(image2, image1, bestH, goodMatches);
        imshow("Overlaying output", overlayingoutput);
        waitKey();
    }
}


int main(int argc, const char * argv[]) {

    Mat input1, input2;
    // Call this once for random number generation
    srand(time(0));
    input1 = imread("image_2.jpg", 1);
    input2 = imread("image_5.jpg", 1);
    run_SIFT(input1,input2);

    return 0;

}

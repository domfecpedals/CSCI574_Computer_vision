//
//  reconstruction.cpp
//  reconstruction
//
//  Created by Yuan Liu on 11/2/15.
//  Copyright © 2015 Yuan Liu. All rights reserved.
//
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include "points.h"

using namespace cv;
using namespace std;

//#	Image1		Image2		Image3		Image4		Image5		Image6
//1	(227, 341)	(150, 208)	(27, 225)	(48, 33)	(85, 93)	(227, 52)
//2	(261, 400)	(112, 197)	(67, 225)	(111, 55)	(126, 97)	(253, 52)
//3	(339, 277)	(192, 83)	(133, 126)	(130, 71)	(235, 133)	(81, 57)
//4	(299, 218)	(234, 88)	(95, 123)	(62, 46)	(197, 131)	(66, 57)
//5	(194, 341)	(175, 228)	(48, 247)	(38, 64)	(76, 126)	(228, 93)
//6	(227, 402)	(137, 220)	(86, 246)	(102, 86)	(117, 130)	(253, 95)
//7	(303, 275)	(214, 105)	(156, 147)	(117, 105)	(227, 168)	(85, 102)
//8	(265, 214)	(256, 111)	(117, 145)	(48, 81)	(188, 166)	(67, 100)


void reconstruct(){
    
    
    int totalImage = 6 ;
    int pointsPerImage = 8;
    
    Mat SRC (Size(8,12),CV_64F);
    
    // Fill in x and y coordinates
    for (int i=0;i<(totalImage*2);i=i+2){
        for (int j=0;j<8;j++){
            SRC.at<double>(i,j)=x_cor[i/2*8+j];
            cout<<SRC.at<double>(i,j)<<" ";
        }
        cout<<endl;
        for (int j=0;j<8;j++){
            SRC.at<double>(i+1,j)=y_cor[i/2*8+j];
            
            cout<<SRC.at<double>(i+1,j)<<" ";
        }
        cout<<endl;
        
    }
    
    // Following 2 steps are to move the coordinate system to center
    // ****************************************
    // calculate xy center coordinates
    Mat centerCoordinates (Size(1,12),CV_64F);
    for (int i = 0 ; i < (totalImage*2) ; i++)
    {
        Scalar Summation = cv::sum(SRC.row(i));
        centerCoordinates.at<double>(i, 0) = ((double)Summation[0]) / ((double)pointsPerImage);
    }
    // subtract the center coordinates to move the system to origin
    for (int i = 0 ; i < (2*totalImage) ; i++)
    {
        SRC.row(i) -= Mat::ones(Size(pointsPerImage, 1), CV_64F) * (centerCoordinates.at<double>(i, 0));
    }
    // ****************************************
    
    Mat W, U, Vt;
    SVD::compute(SRC, W, U, Vt);
    
    Rect roi1(0, 0, 3, 12);
    Rect roi2(0, 0, 3, 8);
    Mat U_combined(U, roi1);
    
    Mat Vt_transposed;
    transpose(Vt, Vt_transposed);
    Mat Vt1(Vt_transposed, roi2);
    Mat W_combined = (Mat_<double>(3,3) << W.at<double>(0,0), 0, 0, 0,
                      W.at<double>(1,0), 0, 0, 0, W.at<double>(2,0));
    
    // the Sqrt transpose method to compute P
    Mat W_combined_sqrt;
    sqrt(W_combined, W_combined_sqrt);
    Mat Vt1_transposed;
    transpose(Vt1, Vt1_transposed);
    Mat A = U_combined * W_combined_sqrt;
    Mat P = W_combined_sqrt * Vt1_transposed;

    cout << "P = " << endl << P << endl;
}

int main(int argc, const char * argv[]) {
    // insert code here...
    reconstruct();
    return 0;
}

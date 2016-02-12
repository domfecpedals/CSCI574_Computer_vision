//
//  problem2.cpp
//  problem2
//
//  Created by Yuan Liu on 9/23/15.
//  Copyright © 2015 Yuan Liu. All rights reserved.
//

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;



Mat MeanShiftSeg(Mat image, int spactialWindow, int colorWindow){
    
    Mat labImage,shiftedLab,outputImage,markerMask,img,imgGray;
    // Convert to CIELAB color space
    cvtColor(image, labImage, CV_BGR2Lab);
    pyrMeanShiftFiltering(labImage, shiftedLab, spactialWindow, colorWindow, 1);
    // Convert back to RGB color space
    cvtColor(shiftedLab, outputImage, CV_Lab2BGR);
    
    return outputImage;
}

int main(int argc, const char * argv[]) {
    
    string filename="317080.jpg";
    int spatial_radius=30;
    int intensity_radius=30;
    
    
    Mat img=imread(filename);
    Mat meanShiftedImg=MeanShiftSeg(img,spatial_radius,intensity_radius);
    namedWindow ("Segmented Image",WINDOW_AUTOSIZE);
    imshow ("Segmented Image",meanShiftedImg);
//    imwrite("spatial"+to_string(spatial_radius)+"_intensity"+to_string(intensity_radius)+"_"+filename, meanShiftedImg);
    waitKey();
    
    return 0;
}

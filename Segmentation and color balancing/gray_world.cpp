//
//  problem1.cpp
//  hw2_opencv
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

// Handle values out of boundary issue
float clip(float input){
    if(input>255){
        return 255;
    }
    return input;
}

// Gray world implementation
Mat gray_world(Mat input_img)
{
    
    Mat dst;
    // initial the output image container with the same dimension with the input image.
    input_img.copyTo(dst);
    
    // Set up iterator to run through the whole image
    cv::Mat_<cv::Vec3b>::iterator iter= input_img.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator iter_end= input_img.end<cv::Vec3b>();
    // Iterator for output Mat
    cv::Mat_<cv::Vec3b>::iterator iter_out= dst.begin<cv::Vec3b>();
    
    float b_factor=0;
    float g_factor=0;
    float r_factor=0;
    
    // Iterator the whole image and count sum of values for each channel
    for(int i=0;i<input_img.rows;i++)
    {
        for(int j=0;j<input_img.cols;j++)
        {
            // Sum all values in each channel
            Vec3b pix=input_img.at<cv::Vec3b>(i,j);
            b_factor+=pix.val[0];
            g_factor+=pix.val[1];
            r_factor+=pix.val[2];
            
        }
    }
    
    // Average value of each channel, only if channel1==channel2==channel3, the incident light is white.
    // Otherwise, white balance is required.
    b_factor/=(input_img.cols*input_img.rows);
    g_factor/=(input_img.cols*input_img.rows);
    r_factor/=(input_img.cols*input_img.rows);
    
    // Calculate the scale factor for each channel, normalized to the maximum channel value
    // Other normalized strategies are possible, using this only here for this problem.
    float r=max(max(g_factor,r_factor),b_factor);
    b_factor=r/(b_factor);
    g_factor=r/(g_factor);
    r_factor=r/(r_factor);
    
    // Apply the scale factor to each pixel
    for ( ; iter!= iter_end; ++iter, ++iter_out)
    {
        cv::Vec3b input_pix=*iter,output_pix;
        output_pix.val[0]=clip(input_pix.val[0]*b_factor);
        output_pix.val[1]=clip(input_pix.val[1]*g_factor);
        output_pix.val[2]=clip(input_pix.val[2]*r_factor);
        *iter_out=output_pix;
        
    }
    return dst;
}


int main(int argc, char* argv[]) {
    
    Mat input_img_1=imread("color1.bmp");
    Mat input_img_2=imread("color2.bmp");
    
    Mat dst_img=gray_world(input_img_2);
    
    imshow("gray world",dst_img);
    imwrite("color2_balanced.bmp",dst_img);
    waitKey(0);

}

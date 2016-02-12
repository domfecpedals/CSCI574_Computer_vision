//
//  main.cpp
//  objectRecognition
//
//  Created by Yuan Liu on 11/12/15.
//  Copyright © 2015 Yuan Liu. All rights reserved.
//


#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;
// This index is used to find the valid image file iteratively
int idx=0;
int main(int argc, char* argv[])
{
    // Parameters to be played with
    // Reduced dimentions 128->20
    int pcaTermsCount = 20;
    int const clusterCount = 100 ;
    int neighborCount = 30;
    
    Mat siftFeatures;
    // Total 100 training images of 5 categories
    int siftNum[100];
    int category=0;
    string trainingPath;
    
    //Training
    while (category!=5)
    {
        // Load all training sets of total 5 categories
        switch(category)
        {
            case 0:
                trainingPath = "images/butterfly/train/";
                break;
            case 1:
                trainingPath = "images/car_side/train/";
                break;
            case 2:
                trainingPath = "images/faces/train/";
                break;
            case 3:
                trainingPath = "images/watch/train/";
                break;
            case 4:
                trainingPath = "images/water_lilly/train/";
                break;
        }
        
        // Total 20 training images for each category, read 20 images then pass to the next folder
        int imageCount=0;
        while(imageCount<20)
        {
            string file;
            file=trainingPath+format("image_%04d.jpg",idx);
            idx++;
            cout<<file<<" Loaded!!"<<endl;
            // Load into grayscale image
            Mat grayimg=imread(file,CV_LOAD_IMAGE_GRAYSCALE);

            
            if (!grayimg.empty()){
                imageCount++;
                // Get SIFT feature descriptors
                SIFT sift;
                vector<KeyPoint> keypoints;
                Mat descriptors;
                sift.operator()(grayimg, noArray(), keypoints, descriptors, false);
                siftFeatures.push_back(descriptors);
                // Stack to store all features of total 5 categories
                siftNum[category*20 + imageCount - 1] = descriptors.rows;
            }
        }
        category++;
        idx=0;
    }

    // Prepare the training sets so that they are ready for querying
    // Run priciple component analysis to get PCA SIFT features, to reduce computation intensity
    // PCA will find first x major components which can represent most features of the object.
    Mat covar, mean;
    calcCovarMatrix(siftFeatures, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_64F);
    Mat eigenvalues, eigenvectors;
    eigen(covar, eigenvalues, eigenvectors);
    Mat principalDirections;
    // Get only first x terms of the feature descriptors
    for(int i=0; i<pcaTermsCount; i++)
        principalDirections.push_back(eigenvectors.row(i));
    // Get PCA SIFT descriptors using result from eigen operation
    Mat pcaSift;
    siftFeatures.convertTo(siftFeatures, CV_64F);
    pcaSift = siftFeatures * principalDirections.t();
    pcaSift.convertTo(pcaSift, CV_32F);
    // Run k-means
    Mat clusterCenters;
    Mat labels;
    kmeans(pcaSift, clusterCount, labels, TermCriteria( CV_TERMCRIT_EPS, 100, 0.55), 10, KMEANS_PP_CENTERS, clusterCenters);
    
    // build direction histogram of the trainning sets
    int trainingHist[100][clusterCount] = {0};
    int step = 0;
    for(int i=0; i<100; i++)
    {
        for(int j=0; j<siftNum[i]; j++)
        {
            if(i==0)
                trainingHist[i][labels.at<int>(j)]++;
            else
                trainingHist[i][labels.at<int>(j + step)]++;
        }
        step = step + siftNum[i];
    }
    
    
    // Object Recognition (image querying)
    cout<<"start image query"<<endl;
    int testImageCount=0;
    idx=0;
    category=0;
    int overallPositiveRate=0;
    while(category<5){
        // Total 10 images in each testing set, so jump out if all images are tested.
        int positiveRate=0;
        while(testImageCount<10)
        {
            string file;
            string testPath;
            switch(category)
            {
                case 0:
                    testPath = "images/butterfly/test/";
                    break;
                case 1:
                    testPath = "images/car_side/test/";
                    break;
                case 2:
                    testPath = "images/faces/test/";
                    break;
                case 3:
                    testPath = "images/watch/test/";
                    break;
                case 4:
                    testPath = "images/water_lilly/test/";
                    break;
            }
            // Test image name incrementally.
            file = testPath+format("image_%04d.jpg",idx);
            
            Mat grayTestImg=imread(file,CV_LOAD_IMAGE_GRAYSCALE);

            // If image exists, run complete querying process, if not, pass to next possible image name.
            if(!grayTestImg.empty()){
                cout<<file;
                testImageCount++;
                
                // Get query image's drescriptor
                SIFT sift;
                vector<KeyPoint> keypoints;
                Mat descriptors;
                sift.operator()(grayTestImg, noArray(), keypoints, descriptors, false);
                Mat queryPCASift;
                descriptors.convertTo(descriptors, CV_64F);
                queryPCASift = descriptors * principalDirections.t();
                
                
                //Euclidean
                double **distance;
                distance = new double *[queryPCASift.rows];
                // Calculate the distance re: query image pca feature vs cluster centers
                for(int i=0; i<queryPCASift.rows; i++)
                {
                    distance[i] = new double [clusterCount];
                    for(int j=0; j<clusterCount; j++) // K is the number of cluster centers
                    {
                        distance[i][j] = 0;
                        for(int k=0; k<pcaTermsCount; k++)
                        {
                            distance[i][j] += pow(queryPCASift.at<double>(i,k) - clusterCenters.at<float>(j,k), 2.0);
                        }
                        distance[i][j] = sqrt(distance[i][j]);
                    }
                }
                
                
                // Find shortest distance of each component
                vector<int> shortestDistances;
                for(int i=0; i<queryPCASift.rows; i++)
                {
                    int index=0;
                    for(int j=1; j<clusterCount; j++)
                    {
                        if(distance[i][index]> distance[i][j]){
                            index = j;
                        }
                    }
                    shortestDistances.push_back(index);
                }
                
                // Gnerate query image's histogram
                int testingHist[clusterCount] = {0};
                for(int i=0; i<queryPCASift.rows; i++)
                {
                    testingHist[shortestDistances[i]]++;
                }
                
                // difference between testing descriptor and training sets
                double difference[100] = {0};
                for(int i=0; i<100; i++)
                {
                    for(int j=0; j<clusterCount; j++)
                    {
                        difference[i] += (testingHist[j] - trainingHist[i][j]) * (testingHist[j] - trainingHist[i][j]);
                    }
                    difference[i] = sqrt(difference[i]);
                }
                
                
                Mat dst;
                Mat src(1,100,CV_64F,&difference);
                sortIdx(src,dst,CV_SORT_EVERY_ROW | CV_SORT_ASCENDING );
                // vote to find out the best prediction
                double vote[5] = {0};
                for(int i=0; i<neighborCount; i++)
                {
                    // Vote 5 categories in the array. 0-20, 20-40,40-60, 60-80, 80-100 represents 5 categories respectively.
                    if(dst.at<int>(i) >= 0 && dst.at<int>(i) < 20)
                        vote[0] += 1/difference[dst.at<int>(i)];
                    else if(dst.at<int>(i) >= 20 && dst.at<int>(i) < 40)
                        vote[1] += 1/difference[dst.at<int>(i)];
                    else if(dst.at<int>(i) >= 40 && dst.at<int>(i) < 60)
                        vote[2] += 1/difference[dst.at<int>(i)];
                    else if(dst.at<int>(i) >= 60 && dst.at<int>(i) < 80)
                        vote[3] += 1/difference[dst.at<int>(i)];
                    else
                        vote[4] += 1/difference[dst.at<int>(i)];
                }
                int max=0;
                for(int i=0; i<5; i++)
                {
                    if(vote[max] < vote[i])
                        max=i;
                }
                
                switch(max)
                {
                    case 0:
                        cout<<" is butterfly"<<endl;
                        break;
                    case 1:
                        cout<<" is car side"<<endl;
                        break;
                    case 2:
                        cout<<" is face"<<endl;
                        break;
                    case 3:
                        cout<<" is watch"<<endl;
                        break;
                    case 4:
                        cout<<" is water lilly"<<endl;
                        break;
                }
                // Record positive rate
                if (max==category){
                    positiveRate+=10;
                }
            }
            idx++;
        }
        cout<<"Prediction positive rate is: "<<positiveRate<<"%"<<endl;
        overallPositiveRate+=positiveRate;
        idx=0;
        testImageCount=0;
        category++;
    }
    cout<<"Overall prediction positive rate is: "<<(float)overallPositiveRate/5.0f<<"%"<<endl;
    return 0;
}

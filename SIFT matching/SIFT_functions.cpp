//
//  SIFT_functions.cpp
//  hw2_opencv
//
//  Created by Yuan Liu on 10/16/15.
//  Copyright © 2015 Yuan Liu. All rights reserved.
//

#include "SIFT_functions.hpp"


using namespace std;
using namespace cv;

// Display sift key points on images using opencv function, drawKeypoints
void displayFeatures(image_container image1, image_container image2)
{
    
    Mat output(max(image1.image.size().height,image2.image.size().height), image1.image.size().width+image2.image.size().width, CV_8UC3);
    
    drawKeypoints(image1.image, image1.keypoints, image1.sift_output, Scalar(0,0,255));  // draw features
    drawKeypoints(image2.image, image2.keypoints, image2.sift_output, Scalar(0,0,255));  // draw features
    
    Mat left(output, Rect(0, 0, image1.image.size().width, image1.image.size().height));
    image1.sift_output.copyTo(left);
    Mat right(output, Rect(image1.image.size().width, 0, image2.image.size().width, image2.image.size().height));
    image2.sift_output.copyTo(right);
    
    imshow("SIFT Feature points", output);
    waitKey();
}


// Run through every pair of key points from both target and test images
// And calculated the euclidean distance between 128 dimension features
vector<DMatch> get2D_euclidian_map(image_container image1, image_container image2){
    
    vector< vector<float> > sorted, unsorted;
    for (int i=0; i<image1.keypoints.size(); i++)
    {
        sorted.push_back(vector<float>());
        unsorted.push_back(vector<float>());
    }
    
    vector< DMatch > match_points;
    
    // loop through keypoints1.size
    for (int i=0; i<image1.keypoints.size(); i++)
    {
        for (int j=0; j<image2.keypoints.size(); j++)
        {
            double sum=0;
            // calculate euclidian distance, 128 dimensions
            for(int x=0; x<128; x++)
            {
                sum += (pow((image1.descriptors.at<float>(i,x) - image2.descriptors.at<float>(j,x)), 2.0));
            }
            sorted[i].push_back((float)sqrt(sum));
            unsorted[i] = sorted[i];
        }
        // Sort the score so that we can choose which match to keep or to dump.
        sort(sorted[i].begin(),sorted[i].end());
        int pos = (int)(find(unsorted[i].begin(), unsorted[i].end(), sorted[i][0]) - unsorted[i].begin());
        
        // Drop matches if the second highest score is larger than 80% of the largest store
        if(sorted[i][0]<sorted[i][1]*0.8){
            DMatch point;
            point.queryIdx = i;
            point.trainIdx = pos;
            point.distance = sorted[i][0];
            match_points.push_back(point);
        }
    }
    
    return match_points;
}

// Count how many inliers exist for this particular transform matrix H
// Inliers and outliers are classified using euclidean distance between them in the 2D image plane.
// H with the most inliers will be the best H.
int countTotalInlier(vector<Point2f> train, vector<Point2f> query, Mat H, double thresh)
{
    int totalInliers=0;
    for(int i=0; i<(train.size()); i++)
    {
        
        Mat point(3,1,CV_64FC1,0.0);
        point.at<double>(0,0)=query.at(i).x;
        point.at<double>(1,0)=query.at(i).y;
        point.at<double>(2,0)=1;
        
        //Homo transform
        Mat projectedPoint=H*point;
        double distanceE;
        distanceE=sqrt(pow((projectedPoint.at<double>(0,0)-train.at(i).x),2)+pow((projectedPoint.at<double>(1,0)-train.at(i).y),2));
        
        //Compare distance with threshold, if less, count as inlier otherwise outliers
        if(distanceE <= thresh)
        {
            totalInliers++;
        }
    }
    cout<<"Inliers quantity: "<<totalInliers<<endl;
    return totalInliers;
}


// Run singular value decomposition method to calculate the transform matrix H
// Pseudo-inverse is another method to calculate the matrix
Mat calculateTranformMatrix(vector<Point2f> query, vector<Point2f> train)
{
    Mat points1(3,4,CV_64FC1,0.0);
    Mat points2(3,4,CV_64FC1,0.0);
    Mat H(3,3,CV_64FC1,0.0);
    
    //Data prep
    for(int i=0; i<(query.size()); i++)
    {
        points1.at<double>(0,i)=query[i].x;
        points1.at<double>(1,i)=query[i].y;
        points1.at<double>(2,i)=1;
        
        points2.at<double>(0,i)=train[i].x;
        points2.at<double>(1,i)=train[i].y;
        points2.at<double>(2,i)=1;
    }
    
    // solve linear equations
    solve(points1.t(), points2.t(), H, DECOMP_SVD); //Singular Value method.
    return H.t();
}

// Generate random index for ransac algorithm
// The key here is to check no identical index can be picked up together
vector<int> randomIndex(int length, int count)
{
    vector<int> randomIndex;
    int counter=0;
    while(randomIndex.size()<count){
        int ran = rand() % length; //[O,(length-1)]
        if(counter==0){
            randomIndex.push_back(ran);
            counter++;
            
        }else{
            bool pass=true;
            for(int i=0;i<randomIndex.size();i++){
                // Check all previous stored index, see if duplicate one exist.
                if(randomIndex[i]==ran){
                    pass=false;
                }
            }
            if(pass){  // Only if there's no identical index, add it to the array.
                randomIndex.push_back(ran);
                counter++;
            }
        }
    }
    
    return randomIndex;
}

// Create an output image with test image as base
// Draw the match points on the test image
// Transform the matching key points on target image to test image's coordinates
// Display in another color
// The pair of dots should be very close to each other or overlapped.
Mat displayOverlaying(image_container image1, image_container image2, Mat H, vector<DMatch> good_matches)
{
    Mat image=image1.image.clone();
    Mat points1(3,(int)good_matches.size(),CV_64FC1,0.0);
    Mat points2(3,(int)good_matches.size(),CV_64FC1,0.0);
    
    for(int i=0; i<good_matches.size(); i++)
    {
        // Prep homo representation for calculation
        points2.at<double>(0,i) = image2.keypoints[good_matches[i].trainIdx].pt.x;
        points2.at<double>(1,i) = image2.keypoints[good_matches[i].trainIdx].pt.y;
        points2.at<double>(2,i) = 1;
    }
    // Use CV function solve for batch matrix operation computation
    solve(H, points2, points1);
    // Display original match points on image1 using red dots
    double maxGoodDistance=0;
    for(int i=0; i<good_matches.size(); i++)
    {
        Point2f center = Point2f(image1.keypoints[good_matches[i].queryIdx].pt.x, image1.keypoints[good_matches[i].queryIdx].pt.y);
        circle(image, center, 2, Scalar(0,0,255), 1);
        maxGoodDistance=maxGoodDistance<good_matches[i].distance?good_matches[i].distance:maxGoodDistance;
    }
    
    // Display translated match points from image2 using green dots
    for(int i=0; i<good_matches.size(); i++)
    {
        Point2f center = Point2f(points1.at<double>(0,i), points1.at<double>(1,i));
        circle(image, center, 2, Scalar(0,255,0), 1);
    }
    return image;
}

// The workflow of RANSAC algorithm, some functions were separated as helper functions to make it more organized.
Mat runRANSACAlgorithm(image_container image1, image_container image2, vector<DMatch> matches, bool &matchExist)
{
    vector<Point2f> query(matches.size());
    vector<Point2f> train(matches.size());
    
    for(int i=0;i<(matches.size());i++)
    {
        query[i]=image1.keypoints[matches[i].queryIdx].pt;
        train[i]=image2.keypoints[matches[i].trainIdx].pt;
    }
    

    Mat H;
    Mat Hbest;
    int mostInliers=0;
    
    int match_length=(int)matches.size();
    int requiredPoints=4;
    
    for (int iteration=0;iteration<2000;iteration++)
    {
        vector<int> randVecId=randomIndex(match_length,requiredPoints);
        vector<Point2f> queryH(requiredPoints);
        vector<Point2f> trainH(requiredPoints);
        
        
        // Extract 4 points from all good matches.
        for(int i=0;i<requiredPoints;i++)
        {
            queryH[i]=query[randVecId[i]];
            trainH[i]=train[randVecId[i]];
        }
        
        H = calculateTranformMatrix(queryH, trainH);  // Compute H based 4 random points
        
        double inlinersNumber = countTotalInlier(train, query, H, 3);  // calculate number of inliners
        
        // Track the best performance re: the transformation
        if(inlinersNumber>mostInliers)
        {
            mostInliers=inlinersNumber;
            Hbest=H.clone();
        }

        
    } //Try 2000 random combinations
    
    cout<<mostInliers/(double)match_length<<endl;
    // If there are only few inliers, that means no good match exist, set the flag and pop up error message.
    if(mostInliers/(double)match_length<0.2){
        matchExist=false;
    }else{
        matchExist=true;
    }
    
    return Hbest;
}


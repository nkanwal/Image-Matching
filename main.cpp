//** AMIE: A descriptor for corner points **/
//** Author: Dr. Nadia Kanwal                 **/
//** Date: 30-12-2011                     **/
//** Contact: VASE lab, School of Computer Science  **/
//** and Electronic Engineering, University of Essex **/
//** Reference  
//** Email: nadia.tahseen@gmail.com
Kanwal, N., Bostanci, E., and Clark, A. F, “Matching Corners Using the informative Arc,” accepted for publication, IET Computer Vision, 2013.

Kanwal, N., Bostanci, E., and Clark, A. F., “Describing Corners using the Angle, Mean Intensity and Entropy of Informative Arcs” Electronic Letters, 48(4), 209–210, 2012.

**/

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <list>
#include "Point.h"
#include "SEA.h"
#include "TestClass.h"
#include "Timer.h"


bool areCloseFeatures(double , double , double , double );

using namespace std;

int main()
{

        int w,h,i,j,count;
        const int MAX_CORNERS =900;
		CvPoint2D32f corners[MAX_CORNERS] = {0};
		CvPoint2D32f Framecorners[MAX_CORNERS] = {0};
		int corner_count = MAX_CORNERS;
		double quality_level =0.05;
		double min_distance = 5;
		int eig_block_size = 3;
		int use_harris = false;
        IplImage* gray_frame, *gray_frame_next= 0;
        IplImage* frame , * next_frame  = 0;
        char video_name[80];
        CornerDescriptor list1;
        list<CornerDescriptor> matches;
        list<CornerDescriptor> frame1_corners;
        list<CornerDescriptor> frame2_corners;
        list<CornerDescriptor>::iterator thiscorners;
         list<CornerDescriptor>::iterator list1It;
        list<CornerDescriptor>::iterator list2It;
        SEA sea;
        Timer timer, time1, time2;
        double ransacThreshold = 2.36;
        int numInliers = 0;
        int frame1corners=0;
        int numOutliers = 0;
        IplImage *eig_image,*temp_image;
        CvMat * homographyMatrix = cvCreateMat(3, 3, CV_32F);
        CvMat * sourceMatrix = cvCreateMat(3, 1, CV_32F);
        CvMat * resultMatrix = cvCreateMat(3, 1, CV_32F);
        Point corner;

  sprintf(video_name,"../../Data/biscuit/MOV00706.AVI");
  CvCapture *capture = cvCaptureFromAVI(video_name );
  frame = cvQueryFrame( capture );
  if(!frame)
    goto next;
  count=0;

    gray_frame_next = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U, 1);
    gray_frame = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U, 1);
    eig_image  = cvCreateImage(cvSize(w, h),IPL_DEPTH_32F, 1);
    temp_image = cvCreateImage(cvSize(w, h),IPL_DEPTH_32F, 1);
    w = frame->width;
    h = frame->height;
    sea.readSize(w,h);

  while(1)
  {
    frame1_corners.clear();
    if(count == 0)
    {
        printf("read\n");
        cvCvtColor(frame, gray_frame, CV_BGR2GRAY);

            timer.start();
        cvGoodFeaturesToTrack(gray_frame,eig_image, temp_image,corners,&corner_count,quality_level,min_distance,NULL,eig_block_size,use_harris);
            timer.stop();

   //****************************** Corner Descriptor *************************************
        frame1corners = corner_count;
        time1.start();
        frame1_corners =  sea.getDescriptor(&gray_frame,corners,frame1corners);
        time1.stop();
        printf("First Descriptor time is %d corners is %gms\t for \n",corner_count,time1.getTimeMS());
    }
    else
    {
       gray_frame = gray_frame_next;

       for(list2It = frame2_corners.begin();list2It != frame2_corners.end();list2It++)
        {
            list1.has_des   = list2It->has_des;
            list1.x         = list2It->x;
            list1.y         = list2It->y;
            list1.mt_x      = list2It->mt_x;
            list1.mt_y      = list2It->mt_y;
            list1.inlier    = list2It->inlier;
            for(j=0;j<25;j++)
            {
                list1.SEA[j]    = list2It->SEA[j];
                list1.mt_SEA[j] = list2It->mt_SEA[j];
            }
            frame1_corners.push_back(list1);
        }
        frame1corners = corner_count;
        printf("copied\n");
        //cvReleaseImage(&gray_frame_next);
    }

        next_frame = cvQueryFrame( capture );
        printf("next frame\n");
        if(!next_frame)
         break;

    ///////////////////////////////////////////////////////////Nexr frame processing///////////////////////////////////////////////////////////////////////////

        corner_count = MAX_CORNERS;
        printf("next\n");
        cvCvtColor(next_frame, gray_frame_next, CV_BGR2GRAY);
        cvGoodFeaturesToTrack(gray_frame_next,eig_image,temp_image,Framecorners,&corner_count,quality_level,min_distance,NULL,eig_block_size,use_harris);

        corner_count = frame1corners ;
            time2.start();
        frame2_corners =  sea.getDescriptor(&gray_frame_next,Framecorners,corner_count);
            time2.stop();
        printf("Second Descriptor time is %d corners is %gms\t for \n",corner_count,time2.getTimeMS());        ///////////////////////// Match Frames /////////////////////////////////////
            timer.start();
        matches = SEA::match_descp(frame1_corners,frame2_corners);
            timer.stop();
//      printf("Matching time is %d corners is %gms\t for \n",corner_count,timer.getTimeMS());

        if(matches.size()>4)
        {

            CvMat * sourcePoints = cvCreateMat(2, matches.size(), CV_32F);
            CvMat * destinationPoints = cvCreateMat(2, matches.size(), CV_32F);

            int counter = 0;

            for(thiscorners=matches.begin();thiscorners!=matches.end();thiscorners++)
            {
                cvmSet(sourcePoints, 0,counter, thiscorners->x);
                cvmSet(sourcePoints, 1,counter, thiscorners->y);
                cvmSet(destinationPoints, 0,counter, thiscorners->mt_x);
                cvmSet(destinationPoints, 1,counter, thiscorners->mt_y);
                counter++;
            }
            cvFindHomography(sourcePoints, destinationPoints, homographyMatrix, CV_RANSAC, ransacThreshold);
            numInliers = 0;
            numOutliers = 0;

            for(thiscorners=matches.begin();thiscorners!=matches.end();thiscorners++)
            {
                cvmSet(sourceMatrix, 0, 0, (double) thiscorners->x);
                cvmSet(sourceMatrix, 1, 0, (double)thiscorners->y);
                cvmSet(sourceMatrix, 2, 0, 1.0);

                cvMatMul(homographyMatrix, sourceMatrix, resultMatrix);

                double x = (double)thiscorners->mt_x;
                double y = (double)thiscorners->mt_y;
                double xBar = cvmGet(resultMatrix, 0, 0) / cvmGet(resultMatrix, 2, 0);
                double yBar = cvmGet(resultMatrix, 1, 0) / cvmGet(resultMatrix, 2, 0);
                thiscorners->inlier = false;
                bool close = areCloseFeatures(x, y, xBar, yBar);
                if(close == true)
                {
                    numInliers++;
                    thiscorners->inlier = true;
                }
                else
                {
                    numOutliers++;
                    thiscorners->inlier = false;
                }
            }


        }

        else
        {
          printf("not sufficient matches.................\n");
          goto next;
        }

        printf("------------inliers = %d out of %d\n",numInliers, matches.size());

        SEA::display_matches(gray_frame, gray_frame_next, matches,frame1_corners,frame2_corners,count);
        count++;


    next:
            printf("------------inliers = %d out of %d\n",numInliers, matches.size());
}


    cvReleaseCapture(&capture);
    cvReleaseImage(&eig_image);
    cvReleaseImage(&temp_image);
    cvReleaseImage(&gray_frame_next);
    cvReleaseImage(& next_frame);
    cvReleaseImage(&frame);
    cvReleaseImage(&gray_frame);

    return 0;
}

bool areCloseFeatures(double x1, double y1, double x2, double y2)
{
    double distanceThreshold = 10.0;

    double result=0.0;
    result = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
    if(result <= distanceThreshold)
        return true;
    else
        return false;
}


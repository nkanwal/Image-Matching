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

#ifndef SEA_H
#define SEA_H
#include <cvaux.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <stdio.h>
#include <math.h>
#include <list>
#include "Point.h"
#include "pthread.h"
#include <omp.h>

#define rad3d 16
#define rad5d 28
#define rad7d 40
#define rad9d 52
#define rad11d 64
#define edgeThresh 0.01
#define diff_thresh 30
#define CHUNKSIZE   100


using namespace std;

class SEA
{
    public:
        SEA();
        void readSize(int ,int );
        list<CornerDescriptor> getDescriptor(IplImage **, CvPoint2D32f * , int );
        void circularStructure();
        void eigenValsVects();
        void OpenCveigenValsVects();
        void detectOrientation();
        void calculateDescriptor();
        void ent_for_descp(int);
        void Ang_Avg_descp(int);
        static list<CornerDescriptor> match_descp(list<CornerDescriptor> , list<CornerDescriptor> );
       static void display_matches(IplImage *,IplImage*,list<CornerDescriptor>,list<CornerDescriptor>,list<CornerDescriptor>,int );
        static IplImage *stack_imgs(IplImage *, IplImage *);



    protected:
    private:

//variables for getCorners

int * rad3C, *rad5C, *rad7C, *rad9C, *rad11C;
IplImage *dst_imgX, *dst_imgY, *conv;
int width, height;
IplImage *GrayImage;
IplImage *displayImage;
IplImage *dispImage;
IplImage *eignvec;
int eig_block_size;
CvPoint2D32f cornersDetected;
int chunk, blocksize;


vector<Point> corners;
vector<Point>::iterator cornersIt;
list<CornerDescriptor> frame_corners;


};

#endif // SEA_H

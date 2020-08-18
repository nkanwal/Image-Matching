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


#include "SEA.h"
#include "Timer.h"





SEA::SEA()
{

    int rad3Ct[32]   = {-3,0,-3,-1,-2,-2,-1,-3,0,-3,1,-3,2,-2,3,-1,3,0,3,1,2,2,1,3,0,3,-1,3,-2,2,-3,1};

    int rad5Ct[56]   = {-5,0,-5,-1,-5,-2,-4,-3,-3,-4,-2,-5,-1,-5,0,-5,1,-5,2,-5,3,-4,4,-3,5,-2,5,-1,5,
                        0,5,1,5,2,4,3,3,4,2,5,1,5,0,5,-1,5,-2,5,-3,4,-4,3,-5,2,-5,1};
    int rad7Ct[80]   = {-7,0,-7,-1,-7,-2,-6,-3,-6,-4,-5,-5,-4,-6,-3,-6,-2,-7,-1,-7,0,-7,1,-7,2,-7,3,-6,4,-6,
                         5,-5,6,-4,6,-3,7,-2,7,-1,7,0,7,1,7,2,6,3,6,4,5,5,4,6,3,6,2,7,1,7,0,7,-1,7,-2,7,-3,6,
                         -4,6,-5,5,-6,4,-6,3,-7,2,-7,1};
    int rad9Ct[104]  = {-9,0,-9,-1,-9,-2,-8,-3,-8,-4,-7,-5,-7,-6,-6,-7,-5,-7,-4,-8,-3,-8,-2,-9,-1,-9,0,
                        -9,1,-9,2,-9,3,-8,4,-8,5,-7,6,-7,7,-6,7,-5,8,-4,8,-3,9,-2,9,-1,9,0,9,1,9,2,8,3,8,
                        4,7,5,7,6,6,7,5,7,4,8,3,8,2,9,1,9,0,9,-1,9,-2,9,-3,8,-4,8,-5,7,-6,7,-7,6,-7,5,-8,
                        4,-8,3,-9,2,-9,1};
    int rad11Ct[128] = {-11,0,-11,-1,-11,-2,-11,-3,-10,-4,-10,-5,-9,-6,-8,-7,-8,-8,-7,-8,-6,-9,-5,-10,
                        -4,-10,-3,-11,-2,-11,-1,-11,0,-11,1,-11,2,-11,3,-11,4,-10,5,-10,6,-9,7,-8,8,-8,8,
                        -7,9,-6,10,-5,10,4,11,-3,11,-2,11,-1,11,0,11,1,11,2,11,3,10,4,10,5,9,6,8,7,8,8,7,
                        8,6,9,5,10,4,10,3,11,2,11,1,11,0,11,-1,11,-2,11,-3,11,-4,10,-5,10,-6,9,-7,8,-8,8,
                        -8,7,-9,6,-10,5,-10,4,-11,3,-11,2,-11,1};

        rad3C  = (int *)malloc(32*sizeof(int));
        rad5C  = (int *)malloc(56*sizeof(int));
        rad7C  = (int *)malloc(80*sizeof(int));
        rad9C  = (int *)malloc(104*sizeof(int));
        rad11C = (int *)malloc(128*sizeof(int));

        for(int i=0; i<32;i++)
            rad3C[i] = rad3Ct[i];

        for(int i=0; i<56;i++)
            rad5C[i] = rad5Ct[i];

        for(int i=0; i<80;i++)
            rad7C[i] = rad7Ct[i];

        for(int i=0; i<104;i++)
            rad9C[i] = rad9Ct[i];

        for(int i=0; i<128;i++)
            rad11C[i] = rad11Ct[i];

        width  = 640;
        height = 480;


    dst_imgX = NULL;
    dst_imgY = NULL;
    conv = NULL;
    eignvec = NULL;
    readSize(width,height);
    blocksize = 3;
	chunk = CHUNKSIZE;
	omp_set_num_threads( 4);

}

void SEA::readSize(int w,int h)
{
    width  = w;
    height = h;
    eig_block_size = 3;
	displayImage    = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,3);
	dispImage       = cvCreateImage(cvSize(displayImage->width*20/100,displayImage->height*20/100), IPL_DEPTH_8U, 3);
	eignvec         = cvCreateImage(cvSize(width*6,height),IPL_DEPTH_32F,1);
}

list<CornerDescriptor> SEA::getDescriptor(IplImage **greyImage, CvPoint2D32f *cornersList, int noCorners)
{
    GrayImage = *greyImage;
    cvMerge(*greyImage,*greyImage,*greyImage,NULL,displayImage);

    Timer timer;
    int i;
    Point corner;
    corners.clear();
    for(i=0; i< noCorners; i++)
    {
         if(cornersList[i].x < GrayImage->width-12 && cornersList[i].x > 12 && cornersList[i].y < GrayImage->height-12 &&  cornersList[i].y > 12 )
	         corner.des = 1;
	   	 else
             corner.des =0;

         corner.x = (int)cornersList[i].x;
         corner.y = (int)cornersList[i].y;
         corner.mt_x = NULL;
         corner.mt_y = NULL;
         corners.push_back(corner);
    }
    circularStructure();  // call function to fill circular templates
//  timer.start();
//  eigenValsVects();      // calculate eigen values in 11x11 patche
    OpenCveigenValsVects(); //calculate eigen values for whole image
//  timer.stop();

//  timer.start();
    detectOrientation();
    calculateDescriptor();
//  timer.stop();
    CornerDescriptor cornerdesc;
    frame_corners.clear();
    for(cornersIt = corners.begin(); cornersIt != corners.end(); cornersIt++)
    {
      if(cornersIt->des == 1)
      {
        cornerdesc.x = cornersIt->x;
        cornerdesc.y = cornersIt->y;
        cornerdesc.has_des = cornersIt->des;
        for(i=0; i<=15;i++)
        {
           cornerdesc.SEA[i] = cornersIt->SEA[i];
        }
           frame_corners.push_back(cornerdesc);

       }
     }

   return frame_corners;
}

void SEA::circularStructure()
{
    int index, rad_index;
    unsigned int i;
#pragma omp parallel for private(rad_index,index)
    for(i= 0; i < corners.size(); i++)
    {
      for(rad_index = 0; rad_index< rad3d; rad_index++)
      {
           index = rad_index*2;
           corners[i].radius3[rad_index].x = corners[i].x+ rad3C[index ];
           corners[i].radius3[rad_index].y = corners[i].y+ rad3C[index + 1];
      }


      for(rad_index = 0; rad_index< rad5d; rad_index++)
      {
         index = rad_index*2;
         corners[i].radius5[rad_index].x = corners[i].x+ rad5C[index ];
         corners[i].radius5[rad_index].y = corners[i].y+ rad5C[index + 1];
      }

      for(rad_index = 0; rad_index< rad7d; rad_index++)
      {
          index = rad_index*2;
          corners[i].radius7[rad_index].x = corners[i].x+ rad7C[index ];
          corners[i].radius7[rad_index].y = corners[i].y+ rad7C[index + 1];
      }

      for(rad_index = 0; rad_index< rad9d; rad_index++)
      {
          index = rad_index*2;
          corners[i].radius9[rad_index].x = corners[i].x+ rad9C[index ];
          corners[i].radius9[rad_index].y = corners[i].y+ rad9C[index + 1];
      }

      for(rad_index = 0; rad_index< rad11d; rad_index++)
      {
        index = rad_index*2;
        corners[i].radius11[rad_index].x    = corners[i].x+ rad11C[index ];
        corners[i].radius11[rad_index].y    = corners[i].y+ rad11C[index + 1];
      }

      }
}

void SEA::OpenCveigenValsVects()
{
    cvCornerEigenValsAndVecs(GrayImage,eignvec, eig_block_size,3);
}

using namespace cv;

void SEA::eigenValsVects()
{
    Mat Dx,Dy;
    int block_size = 3;
    int aperture_size = 3;
    int borderType=BORDER_DEFAULT;
    float* covdata;
    int i,j, ii,jj,l,m;
    int templateXmin,templateXmax;
    int templateYmin,templateYmax;
    int corner_index;
    double  a, b,c,l1,l2,u,v;
    float dx,dy;
    float* cov_data;
    const float* dxdata ,* dydata ;

    cv::Mat src = cv::cvarrToMat(GrayImage);

    int depth = src.depth();
    double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
    if( aperture_size < 0 )
        scale *= 2.;
    if( depth == CV_8U )
        scale *= 255.;
    scale = 1./scale;
    Size size = cvSize(23,23);
   #pragma omp parallel
    {
     #pragma omp sections nowait
      {
      #pragma omp section
        Sobel( src, Dx, CV_32F, 1, 0, aperture_size, scale, 0, borderType );
      #pragma omp section
        Sobel( src, Dy, CV_32F, 0, 1, aperture_size, scale, 0, borderType );

      }
    }

#pragma omp parallel for private(templateXmin,templateXmax,templateYmin,templateYmax,i,j,m,l,dx,dy,ii,jj,a,b,c,u,v,l1,l2,cov_data,covdata,dxdata,dydata)
    for(corner_index = 0; corner_index < corners.size(); corner_index++)
	{
       Mat cov( size, CV_32FC3 );
	   if(corners[corner_index].des == 1)
        {
            templateXmin = corners[corner_index].x - 11;
            templateXmax = corners[corner_index].x + 11;
            templateYmin = corners[corner_index].y - 11;
            templateYmax = corners[corner_index].y + 11;

            l=0;
            for(i= templateYmin   ; i<= templateYmax    ; i++)
            {
              cov_data = (float*)(cov.data + l*cov.step);
              dxdata = (const float*)(Dx.data + i*Dx.step);
              dydata = (const float*)(Dy.data + i*Dy.step);
              m=0;
              for(j = templateXmin  ; j<= templateXmax ; j++)
              {
                dx= dxdata[j];
				dy= dydata[j];
				cov_data[m*3]= dx*dx;
                cov_data[m*3+1]= dx*dy;
				cov_data[m*3+2]= dy*dy;
				m++;
               }
               l++;
            }

           boxFilter(cov, cov, cov.depth(), Size(block_size, block_size),
                        cvPoint(-1,-1), false, borderType );
           ii=0;
           for( i = 0; i < size.height; i++ )
            {
              jj=0;
              covdata = ( float*)(cov.data + cov.step*i);
              for( j = 0; j < size.width; j++ )
              {
                a = covdata[j*3];
                b = covdata[j*3+1];
                c = covdata[j*3+2];
                u  = (a + c)*0.5;
                v  = std::sqrt((a - c)*(a - c)*0.25 + b*b);
                l1 = u + v;
                l2 = u - v;
                corners[corner_index].eigenValsVes[ii][jj].L1= (l1);
                corners[corner_index].eigenValsVes[ii][jj].L2= (l2);
                jj++;
               }
                ii++;
            }

        } //end if

	} // end corners loop

}

void SEA::detectOrientation()
{

    int i,j, size=1,x,y, r, voteSum, WvoteSum;
    int first_index, second_index ;
    CvScalar L1,L2; //double L1,L2;//
    int ii,jj;
    unsigned int corner_index;
    double sum,AvgCornerInt,diff;
    double CornerInt;
    double pixVal;

#pragma omp parallel for private(AvgCornerInt,CornerInt,first_index,second_index,x,y, i,j,L1,L2, ii,jj,diff,sum,r, voteSum, WvoteSum, pixVal)
    for(corner_index= 0; corner_index < corners.size(); corner_index++)
    {
        AvgCornerInt = 0;
        for(r=0;r<5;r++){
            corners[corner_index].vote[r] =0;
            corners[corner_index].Wvote[r] =0;
        }

        if(corners[corner_index].des == 1)
        {
        //-------Average Corner Intensity in 3x3 patch---------------
		for(x= corners[corner_index].x-size; x <= corners[corner_index].x+size; x++)
            for(y= corners[corner_index].y-size; y <= corners[corner_index].y+size; y++)
            {
                CornerInt = ((uchar *)(GrayImage->imageData + y*GrayImage->widthStep))[x];
                AvgCornerInt +=  CornerInt;
			}

        AvgCornerInt = AvgCornerInt / pow(size*2+1,2.0);
        L1 = cvGet2D(eignvec,corners[corner_index].y,corners[corner_index].x*6);//corners[corner_index].eigenValsVes[11+rad3C[ii]][11+rad3C[jj]].L1;//
		L2 = cvGet2D(eignvec,corners[corner_index].y,corners[corner_index].x*6+1);//corners[corner_index].eigenValsVes[11+rad3C[ii]][11+rad3C[jj]].L2;//
		diff = L1.val[0]  - L2.val[0] ;
        first_index = -1;	second_index = -1;
        //--------------find edge pixels on circular arc radius 3---------------------
        for(i=0; i < rad3d; i++)
        {
           ii = i*2;
           jj = i*2+1;
           L1 = cvGet2D(eignvec,corners[corner_index].radius3[i].y,corners[corner_index].radius3[i].x*6);//corners[corner_index].eigenValsVes[11+rad3C[ii]][11+rad3C[jj]].L1;//
		   L2 = cvGet2D(eignvec,corners[corner_index].radius3[i].y,corners[corner_index].radius3[i].x*6+1);//corners[corner_index].eigenValsVes[11+rad3C[ii]][11+rad3C[jj]].L2;//
           diff = L1.val[0]  - L2.val[0] ;
		   if(diff> edgeThresh)
		   {
			 corners[corner_index].radius3[i].edge = 1;
  			 if(first_index == -1)
				first_index = i;
             else
				second_index = i;
			}
        }
		corners[corner_index].edge_index[0] = first_index;
        corners[corner_index].edge_index[1] = second_index;
		j=0; sum = 0.0;
		//select two edge points on radius 3 and vote for direction
		if(first_index != -1 && second_index!= -1  )
		{
            for(i=first_index ; i<= second_index; i++)
            {
                pixVal = ((uchar *)(GrayImage->imageData + corners[corner_index].radius3[i].y*GrayImage->widthStep))[corners[corner_index].radius3[i].x];
                sum = sum + pixVal;
                if(corner_index == 0)
                j++;
            }
            corners[corner_index].sum_des[0] = sum;
			if(abs((sum/(j+1)) - AvgCornerInt) < diff_thresh) //clock-wise
			 corners[corner_index].vote[0] = 1;
			else
			 corners[corner_index].vote[0] = 0;
        }
		else    //select whole circle
		 corners[corner_index].Wvote[0] = 1;

		//----------find edge pixels on circular arc radius 5----------
		first_index = -1;	second_index = -1;
        for(i=0; i < rad5d; i++)
        {
           ii = i*2;
           jj = i*2+1;
           L1 = cvGet2D(eignvec,corners[corner_index].radius5[i].y,corners[corner_index].radius5[i].x*6);//corners[corner_index].eigenValsVes[11+rad5C[ii]][11+rad5C[jj]].L1;//
           L2 = cvGet2D(eignvec,corners[corner_index].radius5[i].y,corners[corner_index].radius5[i].x*6+1);//corners[corner_index].eigenValsVes[11+rad5C[ii]][11+rad5C[jj]].L2;//
           diff = L1.val[0]  - L2.val[0] ;
           if(diff> edgeThresh)
           {
			 corners[corner_index].radius5[i].edge = 1;
 			 if(first_index == -1)
				first_index = i;
			else
				second_index = i;
            }
         }
		corners[corner_index].edge_index[2] = first_index;
		corners[corner_index].edge_index[3] = second_index;
		j=0; sum = 0.0;
		//select two edge points on radius 5 and vote for direction
        if(first_index != -1 && second_index!= -1)
        {
            for(i=first_index ; i<= second_index; i++)
            {
				pixVal = ((uchar *)(GrayImage->imageData + corners[corner_index].radius5[i].y*GrayImage->widthStep))[corners[corner_index].radius5[i].x];
				sum = sum + pixVal;
				j++;
			}
            corners[corner_index].sum_des[1] = sum;
			if(abs((sum/(j+1)) - AvgCornerInt) < diff_thresh) //clock-wise
				 corners[corner_index].vote[1] = 1;
			else
				 corners[corner_index].vote[1] = 0;
         }
		else //select whole circle
			corners[corner_index].Wvote[1] = 1;

        //----------find edge pixels on circular arc radius 7----------
        first_index = -1;	second_index = -1;
        for(i=0; i < rad7d; i++)
        {
            ii = i*2;
            jj = i*2+1;
            L1 = cvGet2D(eignvec,corners[corner_index].radius7[i].y,corners[corner_index].radius7[i].x*6);//corners[corner_index].eigenValsVes[11+rad7C[ii]][11+rad7C[jj]].L1;//
            L2 = cvGet2D(eignvec,corners[corner_index].radius7[i].y,corners[corner_index].radius7[i].x*6+1);//corners[corner_index].eigenValsVes[11+rad7C[ii]][11+rad7C[jj]].L2;//
			diff = L1.val[0]  - L2.val[0] ;
            if(diff> edgeThresh)
            {
                corners[corner_index].radius7[i].edge = 1;
				if(first_index == -1)
                    first_index = i;
				else
					second_index = i;
			}
         }
		corners[corner_index].edge_index[4] = first_index;
		corners[corner_index].edge_index[5] = second_index;
        //select two edge points on radius 7 and vote for direction
        if(first_index != -1 && second_index!= -1 )
        {
            for(i=first_index ; i<= second_index; i++)
            {
                pixVal = ((uchar *)(GrayImage->imageData + corners[corner_index].radius7[i].y*GrayImage->widthStep))[corners[corner_index].radius7[i].x];
				sum = sum + pixVal;
				j++;
			}
            corners[corner_index].sum_des[2] = sum;
			if(abs((sum/(j+1)) - AvgCornerInt) < diff_thresh) //clock-wise
				 corners[corner_index].vote[2] = 1;
			else
                corners[corner_index].vote[2] = 0;
        }
		else //select whole circle
			corners[corner_index].Wvote[2] = 1;

            //----------find edge pixels on circular arc radius 9----------
        first_index = -1;	second_index = -1;
        for(i=0; i < rad9d; i++)
        {
            ii = i*2;
            jj = i*2+1;
            L1 = cvGet2D(eignvec,corners[corner_index].radius9[i].y,corners[corner_index].radius9[i].x*6);//corners[corner_index].eigenValsVes[11+rad9C[ii]][11+rad9C[jj]].L1;//
            L2 = cvGet2D(eignvec,corners[corner_index].radius9[i].y,corners[corner_index].radius9[i].x*6+1);//corners[corner_index].eigenValsVes[11+rad9C[ii]][11+rad9C[jj]].L2;//
			diff = L1.val[0]  - L2.val[0] ;
			if(diff> edgeThresh)
			{
                corners[corner_index].radius9[i].edge = 1;
				if(first_index == -1)
					first_index = i;
				else
					second_index = i;
                }
            }
			corners[corner_index].edge_index[6] = first_index;
			corners[corner_index].edge_index[7] = second_index;
			j=0; sum = 0.0;
            //select two edge points on radius 9 and vote for direction
            if(first_index != -1 && second_index!= -1 )
            {
				for(i=first_index ; i<= second_index; i++)
				{
					pixVal = ((uchar *)(GrayImage->imageData + corners[corner_index].radius9[i].y*GrayImage->widthStep))[corners[corner_index].radius9[i].x];
					sum = sum + pixVal;
					j++;
				}
                corners[corner_index].sum_des[3] = sum;
				if(abs((sum/(j+1)) - AvgCornerInt) < diff_thresh) //clock-wise
				 corners[corner_index].vote[3] = 1;
				else
				 corners[corner_index].vote[3] = 0;
            }
			else //select whole circle
				corners[corner_index].Wvote[3] = 1;

            //----------find edge pixels on circular arc radius 11----------
            first_index = -1;	second_index = -1;
            for(i=0; i < rad11d; i++)
            {
               ii = i*2;
               jj = i*2+1;
               L1 = cvGet2D(eignvec,corners[corner_index].radius11[i].y,corners[corner_index].radius11[i].x*6);//corners[corner_index].eigenValsVes[11+rad11C[ii]][11+rad11C[jj]].L1;//
               L2 = cvGet2D(eignvec,corners[corner_index].radius11[i].y,corners[corner_index].radius11[i].x*6+1);//corners[corner_index].eigenValsVes[11+rad11C[ii]][11+rad11C[jj]].L2;//
			   diff = L1.val[0]  - L2.val[0] ;
               if(diff> edgeThresh)
               {
                    corners[corner_index].radius11[i].edge = 1;
					if(first_index == -1)
						first_index = i;
					else
						second_index = i;
				}
              }
              corners[corner_index].edge_index[8] = first_index;
			  corners[corner_index].edge_index[9] = second_index;
   			  j=0; sum = 0.0;
              //select two edge points on radius 11 and vote for direction
              if(first_index != -1 && second_index!= -1)
              {
				for(i=first_index ; i<= second_index; i++)
				{
					pixVal = ((uchar *)(GrayImage->imageData + corners[corner_index].radius11[i].y*GrayImage->widthStep))[corners[corner_index].radius11[i].x];
					sum = sum + pixVal;
					j++;
				}
				corners[corner_index].sum_des[4] = sum;
				if(abs((sum/(j+1)) - AvgCornerInt) < diff_thresh) //clock-wise
					 corners[corner_index].vote[4] = 1;
				else
					 corners[corner_index].vote[4] = 0;
               }
			   else //select whole circle
                 corners[corner_index].Wvote[4] = 1;
            }

            voteSum =0;WvoteSum =0;
            for(r=0; r <5; r++)
            {
                voteSum +=  corners[corner_index].vote[r];
                WvoteSum +=  corners[corner_index].Wvote[r];
            }
			if(WvoteSum < 3)
			{
               if(voteSum >= 3)
                   corners[corner_index].direction = 1; //clockwise
               else
                   corners[corner_index].direction = 2; //anti-clockwise
            }
            else
               corners[corner_index].direction = 3;   //whole circle

        }//end loop for all corners
}// end main

void SEA::calculateDescriptor()
{
   int corner_index;
#pragma omp parallel for
    for(corner_index= 0; corner_index < corners.size(); corner_index++)
    {
        //Calculate sum and then average; calculate Entropy; Add them to descriptor
        if(corners[corner_index].des == 1)
        {
           corners[corner_index].SEA[15] = corners[corner_index].direction;
           Ang_Avg_descp(corner_index);
           ent_for_descp(corner_index) ;
        }
    }

}

void SEA::Ang_Avg_descp(int no)
{
    double sum=0;
    int first_index=0, second_index=0;
    int j , i, k, index = -1,r, circle=0, next_index=0;
    double pixVal;
    CvScalar s;
    int circleMaping[5] = {15,27,39,51,63};

    if(corners[no].direction == 1) //clockkwise
    {
       for(r = 0; r<=4 ; r++)
       {
          if( corners[no].vote[r] == 1)
             circle = r;
       }
       for(k=0;k < 10; k++)
       {
           if(corners[no].edge_index[k] == -1)
           {
              index = k;
              next_index = circle*2;
              if(index % 2 == 1 )
                next_index++;
            corners[no].edge_index[index] = (int)((double)(corners[no].edge_index[next_index] * (circleMaping[index/2])) / (double) (circleMaping[circle]));
            if(index%2 == 1 )
            {
              first_index = index-1;
              second_index = index;
            }
            else
            {
              first_index = index;
              second_index = index+1;
            }
            for(i=first_index ; i<= second_index; i++)
            {
               if(index/2 == 0)
                 pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius3[i].y*GrayImage->widthStep))[corners[no].radius3[i].x];
                 if(index/2 == 1)
                    pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius5[i].y*GrayImage->widthStep))[corners[no].radius5[i].x];
                    if(index/2 == 2)
                       pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius7[i].y*GrayImage->widthStep))[corners[no].radius7[i].x];
                       if(index/2 == 3)
                          pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius9[i].y*GrayImage->widthStep))[corners[no].radius9[i].x];
                          if(index/2 == 4)
                             pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius11[i].y*GrayImage->widthStep))[corners[no].radius11[i].x];
			sum = sum + pixVal;
            corners[no].sum_des[index/2] = sum;
            }

           }
       }
       corners[no].SEA[0] = corners[no].edge_index[1] -corners[no].edge_index[0] +1;
       corners[no].SEA[5] = corners[no].sum_des[0]/corners[no].SEA[0] ;
       corners[no].SEA[1] = corners[no].edge_index[3] -corners[no].edge_index[2] +1;
       corners[no].SEA[6] = corners[no].sum_des[1]/corners[no].SEA[1] ;
       corners[no].SEA[2] = corners[no].edge_index[5] -corners[no].edge_index[4] +1;
       corners[no].SEA[7] = corners[no].sum_des[2]/corners[no].SEA[2] ;
       corners[no].SEA[3] = corners[no].edge_index[7] -corners[no].edge_index[6] +1;
       corners[no].SEA[8] = corners[no].sum_des[3]/corners[no].SEA[3] ;
       corners[no].SEA[4] = corners[no].edge_index[9] -corners[no].edge_index[8] +1;
       corners[no].SEA[9] = corners[no].sum_des[4]/corners[no].SEA[4] ;

    }
    else if(corners[no].direction == 2) //anti-clockwise descriptor
    {
       for(r = 4; r>=0 ; r--)
            if( corners[no].vote[r] == 2)
               circle = r;
       for(k=0;k < 10; k++)
       {
           if(corners[no].edge_index[k] == -1)
           {
              index = k;
              next_index = circle*2;
              if(index % 2 == 1 )
                 next_index++;
              corners[no].edge_index[index] = (int)((double)(corners[no].edge_index[next_index] * (circleMaping[index/2])) / (double) (circleMaping[circle]));
           }
       }
       sum =0;
       corners[no].SEA[0] = (rad3d - corners[no].edge_index[1]) + 1 + corners[no].edge_index[0];
       for(j=corners[no].edge_index[1];j < rad3d;j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius3[j].y*GrayImage->widthStep))[corners[no].radius3[j].x];
           sum += pixVal;
       }
       for(j=0;j<corners[no].edge_index[0];j++)
       {
          pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius3[j].y*GrayImage->widthStep))[corners[no].radius3[j].x];
          sum += pixVal;
       }
       corners[no].SEA[5] =  sum / corners[no].SEA[0]  ;
       sum = 0;
       corners[no].SEA[1] = (rad5d - corners[no].edge_index[3]) + 1 + corners[no].edge_index[2];
       for(j=corners[no].edge_index[3];j < rad5d;j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius5[j].y*GrayImage->widthStep))[corners[no].radius5[j].x];
           sum += pixVal;
       }
       for(j=0;j<corners[no].edge_index[2];j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius5[j].y*GrayImage->widthStep))[corners[no].radius5[j].x];
           sum += pixVal;
       }
       corners[no].SEA[6] = sum / corners[no].SEA[1];
       sum = 0;
       corners[no].SEA[2] =(rad7d - corners[no].edge_index[5] )+ 1 + corners[no].edge_index[4];
       for(j=corners[no].edge_index[5];j < rad7d;j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius7[j].y*GrayImage->widthStep))[corners[no].radius7[j].x];
           sum += pixVal;
       }
       for(j=0;j<corners[no].edge_index[4];j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius7[j].y*GrayImage->widthStep))[corners[no].radius7[j].x];
           sum += pixVal;
       }
       corners[no].SEA[7] = sum / corners[no].SEA[2] ;
       sum = 0;
       corners[no].SEA[3] = (rad9d - corners[no].edge_index[7] )+ 1 + corners[no].edge_index[6];
       for(j=corners[no].edge_index[7];j < rad9d;j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius9[j].y*GrayImage->widthStep))[corners[no].radius9[j].x];
           sum += pixVal;
       }
       for(j=0;j<corners[no].edge_index[6];j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius9[j].y*GrayImage->widthStep))[corners[no].radius9[j].x];
           sum += pixVal;
       }
       corners[no].SEA[8] = sum/ corners[no].SEA[3] ;
       sum = 0;
       corners[no].SEA[4] = (rad11d - corners[no].edge_index[9] )+ 1 + corners[no].edge_index[8];
       for(j=corners[no].edge_index[9];j < rad11d;j++)
       {
          pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius11[j].y*GrayImage->widthStep))[corners[no].radius11[j].x];
          sum += pixVal;
       }
       for(j=0;j<corners[no].edge_index[8];j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius11[j].y*GrayImage->widthStep))[corners[no].radius11[j].x];
           sum += pixVal;
       }
       corners[no].SEA[9] = sum / corners[no].SEA[4] ;
    }
    else    //whole circle descriptor
    {
       corners[no].SEA[0] = rad3d;
       for(j=0;j < rad3d;j++)
       {
          pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius3[j].y*GrayImage->widthStep))[corners[no].radius3[j].x];
          sum += pixVal;
       }
       corners[no].SEA[5] =  sum/ corners[no].SEA[0];
       sum = 0;
       corners[no].SEA[1] = rad5d;
       for(j=0;j<rad5d;j++)
       {
          pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius5[j].y*GrayImage->widthStep))[corners[no].radius5[j].x];
          sum += pixVal;
       }
       corners[no].SEA[6] = sum / corners[no].SEA[1];
       sum = 0;
       corners[no].SEA[2] = rad7d;
       for(j=0;j<rad7d;j++)
       {
            pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius7[j].y*GrayImage->widthStep))[corners[no].radius7[j].x];
            sum += pixVal;
       }
       corners[no].SEA[7] = sum/ corners[no].SEA[2] ;
       sum = 0;
       corners[no].SEA[3] = rad9d;
       for(j=0;j<rad9d;j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius9[j].y*GrayImage->widthStep))[corners[no].radius9[j].x];
           sum += pixVal;
       }
       corners[no].SEA[8] = sum / corners[no].SEA[3] ;
       sum = 0;
       corners[no].SEA[4] = rad11d;
       for(j=0;j<rad11d;j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData + corners[no].radius11[j].y*GrayImage->widthStep))[corners[no].radius11[j].x];
           sum += pixVal;
       }
       corners[no].SEA[9] = sum / corners[no].SEA[4];
    }


}
void SEA::ent_for_descp(int no)
{
    double ent,sum=0.0,logP=0.0;
	int hist[256];
	int i,j,location, first, second;
	double pixVal;

if(corners[no].direction == 1) //clockwise
{

    location = 0;
    first = corners[no].edge_index[location];
    second = corners[no].edge_index[location+1];
    for(j=0;j<256;j++)
        hist[j]=0;
    for(j=first;j<second;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius3[j].y*GrayImage->widthStep))[corners[no].radius3[j].x];
        hist[(int)pixVal]++;
    }
    for(i=0;i<256;i++)
    {
       if(hist[i]!= 0)
       {
          sum += hist[i];
          logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
        }
    }
	if(sum == 0.0)
		ent = 0.0;
    else
	    ent = (log(sum)/log(2.0)) - logP / sum;
    corners[no].SEA[10] = ent;
    location = 2;
    first = corners[no].edge_index[location];
    second = corners[no].edge_index[location+1];
    for(j=0;j<256;j++)
       hist[j]=0;
    for(j=first;j<second;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius5[j].y*GrayImage->widthStep))[corners[no].radius5[j].x];
        hist[(int)pixVal]++;
    }
    for(i=0;i<256;i++)
    {
       if(hist[i]!= 0)
       {
          sum += hist[i];
          logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
       }
    }
	if(sum == 0.0)
        ent = 0.0;
    else
		ent = (log(sum)/log(2.0)) - logP / sum;
    corners[no].SEA[11] = ent;
    location = 4;
    first = corners[no].edge_index[location];
    second = corners[no].edge_index[location+1];
    for(j=0;j<256;j++)
        hist[j]=0;
    for(j=first;j<second;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius7[j].y*GrayImage->widthStep))[corners[no].radius7[j].x];
        hist[(int)pixVal]++;
    }
    for(i=0;i<256;i++)
    {
       if(hist[i]!= 0)
       {
          sum += hist[i];
          logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
       }
    }
	if(sum == 0.0)
        ent = 0.0;
	else
		ent = (log(sum)/log(2.0)) - logP / sum;
	corners[no].SEA[12] = ent;
    location = 6;
    first = corners[no].edge_index[location];
    second = corners[no].edge_index[location+1];
    for(j=0;j<256;j++)
       hist[j]=0;
    for(j=first;j<second;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius9[j].y*GrayImage->widthStep))[corners[no].radius9[j].x];
        hist[(int)pixVal]++;
    }
    for(i=0;i<256;i++)
    {
        if(hist[i]!= 0)
        {
           sum += hist[i];
           logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
        }
    }
	if(sum == 0.0)
		ent = 0.0;
	else
		ent = (log(sum)/log(2.0)) - logP / sum;
	corners[no].SEA[13] = ent;
    location = 8;
    first = corners[no].edge_index[location];
    second = corners[no].edge_index[location+1];
    for(j=0;j<256;j++)
       hist[j]=0;
    for(j=first;j<second;j++)
    {
       pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius11[j].y*GrayImage->widthStep))[corners[no].radius11[j].x];
       hist[(int)pixVal]++;
    }
    for(i=0;i<256;i++)
    {
       if(hist[i]!= 0)
       {
          sum += hist[i];
          logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
        }
     }
	if(sum == 0.0)
        ent = 0.0;
	else
        ent = (log(sum)/log(2.0)) - logP / sum;
    corners[no].SEA[14] = ent;

}

if(corners[no].direction == 2) //anti-clockwise
{
    location = 0;
    first = corners[no].edge_index[location];
    second = corners[no].edge_index[location+1];
    for(j=0;j<256;j++)
       hist[j]=0;
    for(j=second;j<rad3d;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius3[j].y*GrayImage->widthStep))[corners[no].radius3[j].x];
        hist[(int)pixVal]++;
    }
    for(j=0;j<first;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius3[j].y*GrayImage->widthStep))[corners[no].radius3[j].x];
        hist[(int)pixVal]++;
    }
    for(i=0;i<256;i++)
    {
       if(hist[i]!= 0)
       {
          sum += hist[i];
          logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
        }
    }
	if(sum == 0.0)
		ent = 0.0;
	else
		ent = (log(sum)/log(2.0)) - logP / sum;
	corners[no].SEA[10] = ent;
    location = 2;
    first = corners[no].edge_index[location];
    second = corners[no].edge_index[location+1];
    for(j=0;j<256;j++)
        hist[j]=0;
    for(j=second;j<rad5d;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius5[j].y*GrayImage->widthStep))[corners[no].radius5[j].x];
        hist[(int)pixVal]++;
    }
    for(j=0;j<first;j++)
    {
       pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius5[j].y*GrayImage->widthStep))[corners[no].radius5[j].x];
       hist[(int)pixVal]++;
    }
    for(i=0;i<256;i++)
    {
       if(hist[i]!= 0)
       {
          sum += hist[i];
          logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
       }
    }
	if(sum == 0.0)
		ent = 0.0;
    else
		ent = (log(sum)/log(2.0)) - logP / sum;
	corners[no].SEA[11] = ent;
    location = 4;
    first = corners[no].edge_index[location];
    second = corners[no].edge_index[location+1];
    for(j=0;j<256;j++)
       hist[j]=0;
    for(j=second;j<rad7d;j++)
    {
       pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius7[j].y*GrayImage->widthStep))[corners[no].radius7[j].x];
       hist[(int)pixVal]++;
    }
    for(j=0;j<first;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius7[j].y*GrayImage->widthStep))[corners[no].radius7[j].x];
        hist[(int)pixVal]++;
    }
    for(i=0;i<256;i++)
    {
        if(hist[i]!= 0)
        {
           sum += hist[i];
           logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
        }
    }
	if(sum == 0.0)
		ent = 0.0;
	else
		ent = (log(sum)/log(2.0)) - logP / sum;
	corners[no].SEA[12] = ent;
    location = 6;
    first = corners[no].edge_index[location];
    second = corners[no].edge_index[location+1];
    for(j=0;j<256;j++)
        hist[j]=0;
    for(j=second;j<rad9d;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius9[j].y*GrayImage->widthStep))[corners[no].radius9[j].x];
        hist[(int)pixVal]++;
    }
    for(j=0;j<first;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius9[j].y*GrayImage->widthStep))[corners[no].radius9[j].x];
        hist[(int)pixVal]++;
    }
    for(i=0;i<256;i++)
    {
        if(hist[i]!= 0)
        {
            sum += hist[i];
            logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
        }
    }
	if(sum == 0.0)
		ent = 0.0;
	else
		ent = (log(sum)/log(2.0)) - logP / sum;
	corners[no].SEA[13] = ent;
    location = 8;
    first = corners[no].edge_index[location];
    second = corners[no].edge_index[location+1];
    for(j=0;j<256;j++)
       hist[j]=0;
    for(j=second;j<rad11d;j++)
    {
        pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius11[j].y*GrayImage->widthStep))[corners[no].radius11[j].x];
        hist[(int)pixVal]++;
    }
    for(j=0;j<first;j++)
    {
       pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius11[j].y*GrayImage->widthStep))[corners[no].radius11[j].x];
       hist[(int)pixVal]++;
    }
    for(i=0;i<256;i++)
    {
        if(hist[i]!= 0)
        {
           sum += hist[i];
           logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
        }
    }
	if(sum == 0.0)
		ent = 0.0;
	else
		ent = (log(sum)/log(2.0)) - logP / sum;
	corners[no].SEA[14] = ent;
}
else
{
   for(j=0;j<256;j++)
       hist[j]=0;
       for(j=0;j<rad3d;j++)
       {
           pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius3[j].y*GrayImage->widthStep))[corners[no].radius3[j].x];
           hist[(int)pixVal]++;
       }
       for(i=0;i<256;i++)
       {
            if(hist[i]!= 0)
            {
               sum += hist[i];
               logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
            }
       }
		if(sum == 0.0)
            ent = 0.0;
        else
			ent = (log(sum)/log(2.0)) - logP / sum;

		corners[no].SEA[10] = ent;

        for(j=0;j<256;j++)
            hist[j]=0;
        for(j=0;j<rad5d;j++)
        {
           pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius5[j].y*GrayImage->widthStep))[corners[no].radius5[j].x];
           hist[(int)pixVal]++;
        }
        for(i=0;i<256;i++)
        {
            if(hist[i]!= 0)
            {
                sum += hist[i];
                logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
            }
        }
		if(sum == 0.0)
			ent = 0.0;
		else
			ent = (log(sum)/log(2.0)) - logP / sum;

		corners[no].SEA[11] = ent;

        for(j=0;j<256;j++)
            hist[j]=0;
        for(j=0;j<rad7d;j++)
        {
            pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius7[j].y*GrayImage->widthStep))[corners[no].radius7[j].x];
            hist[(int)pixVal]++;
        }
        for(i=0;i<256;i++)
        {
           if(hist[i]!= 0)
           {
              sum += hist[i];
              logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
           }
        }
		if(sum == 0.0)
            ent = 0.0;
        else
            ent = (log(sum)/log(2.0)) - logP / sum;

        corners[no].SEA[12] = ent;

        for(j=0;j<256;j++)
            hist[j]=0;
        for(j=0;j<rad9d;j++)
        {
            pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius9[j].y*GrayImage->widthStep))[corners[no].radius9[j].x];
            hist[(int)pixVal]++;
        }
        for(i=0;i<256;i++)
        {
            if(hist[i]!= 0)
            {
               sum += hist[i];
               logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
            }
        }
		if(sum == 0.0)
			ent = 0.0;
		else
			ent = (log(sum)/log(2.0)) - logP / sum;

		corners[no].SEA[13] = ent;

        for(j=0;j<256;j++)
             hist[j]=0;
        for(j=0;j<rad11d;j++)
        {
            pixVal = ((uchar *)(GrayImage->imageData +corners[no].radius11[j].y*GrayImage->widthStep))[corners[no].radius11[j].x];
            hist[(int)pixVal]++;
        }
        for(i=0;i<256;i++)
        {
            if(hist[i]!= 0)
            {
               sum += hist[i];
               logP += -1*( hist[i]* (log((double)hist[i])/log(2.0)));
            }
        }
		if(sum == 0.0)
			ent = 0.0;
		else
			ent = (log(sum)/log(2.0)) - logP / sum;
		corners[no].SEA[14] = ent;
}

}

list<CornerDescriptor> SEA::match_descp(list<CornerDescriptor> list1, list<CornerDescriptor> list2)
{
    int flag,i;
	int angle_index;
	CornerDescriptor mtp;
	double des_diff, dist,distM, distE, diff,diffM,diffE, ang_diff, maxM, maxE;
	flag = 0;
    list<CornerDescriptor>::iterator list1It;
    list<CornerDescriptor>::iterator list2It;
    list<CornerDescriptor> matches;
    CornerDescriptor matchPoints;
    matches.clear();
    for(list1It = list1.begin(); list1It != list1.end(); list1It++)
    {
        list1It->mt_x = 0;
        list1It->mt_y = 0;
        if(list1It->has_des == 1)
        {

            maxM = maxE = DBL_MAX;

         for(list2It = list2.begin(); list2It != list2.end(); list2It++)
         {
                dist = 0.0; ang_diff=0.0, diff = 0.0;

                for(angle_index = 0;angle_index < 5; angle_index++)
                {
                    ang_diff = (list1It->SEA[angle_index] - list2It->SEA[angle_index]);
                    diff += (ang_diff * ang_diff );
                }
                dist = sqrt(diff);
                //************if angle difference is less than +- 10 and both corners' have similar informative region's direction
                if( dist < 10.0 && (list1It->SEA[15]) == list2It->SEA[15] )
                {

                    flag = 1;
					des_diff =  0.0, diffM = 0.0, distM = 0.0;
                //****************calculate mean intensity differernce***********

					for(i=5;i<10;i++){
						des_diff  = (list1It->SEA[i] - list2It->SEA[i]);
						diffM += (des_diff * des_diff);
                    }
					distM = sqrt(diffM);

                 //****************calculate entropy differernce******************
					des_diff = 0.0, diffE =0.0, distE=0.0;
					for(i=10;i<15;i++){
						des_diff  = (list1It->SEA[i] - list2It->SEA[i]);
						diffE += (des_diff * des_diff);
                    }
                    distE = sqrt(diffE);

					if(distM < maxM && distE < maxE)
					{

						maxM = distM;
						maxE = distE;
						mtp.x = list2It->x;
						mtp.y = list2It->y;

						for(i=0;i<=15;i++)
                            mtp.mt_SEA[i] = list2It->SEA[i];

                        }

                }
        }


           if(maxM < 500.0 && maxE < 5.0)
           {

                    list1It->mt_x = mtp.x;
                    list1It->mt_y = mtp.y;

                    for(i=0;i<=15;i++)
                        list1It->mt_SEA[i] = mtp.mt_SEA[i];

            }
            else{
                    list1It->mt_x = -1;
                    list1It->mt_y = -1;
                }
        }
        else
        {
            list1It->mt_x = -1;
            list1It->mt_y = -1;
        }


    }
    //-----------matching done -------------------------

     for(list1It = list1.begin(); list1It != list1.end(); list1It++)
    {

        if(list1It->mt_x != -1 && list1It->mt_y != -1)
        {

        matchPoints.x = list1It->x;
        matchPoints.y = list1It->y;
        matchPoints.mt_x = list1It->mt_x;
        matchPoints.mt_y = list1It->mt_y;


         for(i=0;i<15;i++)
            matchPoints.mt_SEA[i] =  list1It->mt_SEA[i];
          des_diff = diffE = 0;
            for(i=0;i<15;i++){
                des_diff  = (list1It->SEA[i] - list1It->mt_SEA[i]);
                diffE += (des_diff * des_diff);
            }
        matches.push_back(matchPoints);
        }
    }
    return matches;

}

void SEA::display_matches(IplImage *img1, IplImage *img2, list<CornerDescriptor> cornerslist,list<CornerDescriptor> cornerslist1,list<CornerDescriptor> cornerslist2,int ct)
{


	CvPoint pt1, pt2;
	int count=0;
	char filename[80];
	IplImage * stacked ;
	list<CornerDescriptor>::iterator list1It, list1It2, listd;

	stacked = SEA::stack_imgs( img1, img2 );

	 for(listd= cornerslist1.begin(); listd!= cornerslist1.end(); listd++)
      {

         cvDrawCircle(stacked, cvPoint(listd->x , listd->y ), 2, cvScalar(255, 255, 255),1);
      }


    for(list1It2= cornerslist2.begin(); list1It2!= cornerslist2.end(); list1It2++)
    {
        cvDrawCircle(stacked, cvPoint(list1It2->x, list1It2->y+ img1->height), 2, cvScalar(255, 255, 255), 1);
    }


	 for(list1It= cornerslist.begin(); list1It!= cornerslist.end(); list1It++)
	  {
        if(list1It->inlier == true)
		{

            pt1 = cvPoint(  list1It->x , list1It->y );
            pt2 = cvPoint(  list1It->mt_x , list1It->mt_y );
			printf("%d\t%d\n",list1It->x -list1It->mt_x , list1It->y- list1It->mt_y);
			pt2.y += img1->height;
			cvLine( stacked, pt1, pt2, CV_RGB(255,255,255), 1, 8, 0 );
			count++;
		}
	  }

        //***** if images are too big than uncomment following two lines and display "display" image****************
               // IplImage * display = cvCreateImage(cvSize(stacked->width*25/100,stacked->height*15/100), IPL_DEPTH_8U, 1);
               // cvResize(stacked,display);
                printf("counted inliers %d\n",count);

				cvNamedWindow( "Des_Matches", CV_WINDOW_AUTOSIZE);
				cvShowImage( "Des_Matches", stacked );
				cvWaitKey(10);
}

IplImage *SEA::stack_imgs(IplImage *img1, IplImage *img2)
{

	IplImage* stacked = cvCreateImage( cvSize( MAX(img1->width, img2->width),
										img1->height + img2->height ),
										img1->depth,img1->nChannels );

	cvZero( stacked );
	cvSetImageROI( stacked, cvRect( 0, 0, img1->width, img1->height ) );
	cvAdd( img1, stacked, stacked, NULL );
	cvSetImageROI( stacked, cvRect(0, img1->height, img2->width, img2->height) );
	cvAdd( img2, stacked, stacked, NULL );
	cvResetImageROI( stacked );
 printf("done\n");
	return stacked;

}

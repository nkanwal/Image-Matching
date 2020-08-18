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

#ifndef POINT_H
#define POINT_H
#include "Radius_Info.h"
#include <cv.h>
#include "eigen.h"

#define sum_hist_bin 50

class Point
{
    public:
        Point();

  	int x;							/**< x coord */
	int y;							/**< y coord */
	int mt_x;						/**< x coord */
	int mt_y;						/**< y coord */
	int des;
	int edge_index[10];
	int direction;              /** 1 for clockwise, 2 for anticlockwise, 3 for whole circle **/
	Radius_Info radius3[16];
	Radius_Info radius5[28];
	Radius_Info radius7[40];
	Radius_Info radius9[52];
	Radius_Info radius11[64];
	double SEA[25];
	double sum_des[sum_hist_bin];	/** Sum of inside pixels histogram descriptor*/
	double A_sum_des[sum_hist_bin];
    eigen eigenValsVes[23][23];
    int vote[5];
    int Wvote[5];

    protected:
    private:
};

class CornerDescriptor
{
public:
    int x;
    int y;
    int has_des;
    double SEA[25];
    int mt_x;
    int mt_y;
    bool inlier;
    double mt_SEA[25];

};

#endif // POINT_H

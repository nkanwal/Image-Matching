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

/* 
 * File:   Timer.h
 * Author: gebost
 *
 * Created on 24 September 2010, 11:04
 */

#ifndef TIMER_H
#define	TIMER_H

//#include "GenericHeader.h"
#include<cv.h>
#include<cxcore.h>

class Timer
{
public:
    Timer();
    void start(); // starts the timer
    void stop(); // stops timer
    double getTimeMS(); // returns the time in miliseconds
    double getTimeS(); // returns the time in seconds
private:
    double startTime;
    double stopTime;
    double timing; // difference between start and stop in miliseconds
};

#endif	/* TIMER_H */


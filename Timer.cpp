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

#include "Timer.h"

Timer::Timer()
{
    startTime = 0.0;
    stopTime = 0.0;
    timing = 0.0;
}

void Timer::start()
{
    startTime = (double)cvGetTickCount();
}

void Timer::stop()
{
    stopTime = (double)cvGetTickCount();
}

double Timer::getTimeMS() // returns the time in miliseconds
{
    return timing = (stopTime - startTime) / (cvGetTickFrequency() * 1000.);
}

double Timer::getTimeS() // returns the time in seconds
{
    return timing = getTimeMS()/1000;
}

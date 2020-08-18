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

#ifndef RADIUS_INFO_H
#define RADIUS_INFO_H


class Radius_Info
{
    public:
        Radius_Info();
        int x;							/**< x coord */
        int y;							/**< y coord */
        bool edge;

    protected:
    private:
};

#endif // RADIUS_INFO_H

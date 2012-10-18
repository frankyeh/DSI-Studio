#ifndef TRACKING_METHOD_HPP
#define TRACKING_METHOD_HPP
#include "boost/mpl/vector.hpp"
#include "basic_process.hpp"
#include "roi.hpp"

typedef boost::mpl::vector<
			EstimateNextDirection,
			SmoothDir,
			MoveTrack
> streamline_method_process;

typedef boost::mpl::vector<
            //EstimateNextDirection,
            //SmoothDir,
            //MoveTrack2
            LocateVoxel
> voxel_tracking;


typedef boost::mpl::vector<
			EstimateNextDirectionRungeKutta4,
			MoveTrack
> streamline_runge_kutta_4_method_process;




#endif//TRACKING_METHOD_HPP

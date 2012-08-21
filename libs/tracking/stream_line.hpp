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
			EstimateNextDirection,
			SmoothDir,
			MoveTrack2
> streamline_method_process_with_relocation;


/*
typedef boost::mpl::vector<
			EstimateNextDirectionWithEndpointCheck,
			SmoothDir,
			MoveTrack
> streamline_method_with_endpoint_check_process;
*/

typedef boost::mpl::vector<
			EstimateNextDirectionRungeKutta4,
			MoveTrack
> streamline_runge_kutta_4_method_process;

typedef boost::mpl::vector<
			EstimateNextDirection,
			RecursiveEstimateNextDirection,
			MoveTrack
> second_order_streamline_method_process;


typedef boost::mpl::vector<
			EstimateNextDirectionRungeKutta853
> streamline_adpative_method_process;



/*
typedef boost::mpl::vector<
			MeetROI,
			EstimateNextDirection,
			FaWeightedNextDirection,
			CheckAllowedAngle,
			MoveTrack,
			CheckFaThreshold
> fa_weighted_method_process;
#include "bayesian_direction.hpp"

typedef boost::mpl::vector<
			MeetROI,
			BayesianDirection,
			SmoothDir,
			MoveTrack
> bayesian_method_process;
*/




typedef boost::mpl::vector<
		    RandomDirection
> random_direction_seeding;

typedef boost::mpl::vector<
		    MainFiberDirection
> main_fiber_direction_seeding;



#endif//TRACKING_METHOD_HPP

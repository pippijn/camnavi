#ifndef _REC_MATH_UTILS_H_
#define _REC_MATH_UTILS_H_

#include "rec/math/defines.h"
#include <cmath>

namespace rec
{
  namespace math
  {
		static Real deg2rad( Real deg )
		{
			return PI * deg / 180.0f;
		}

		static Real rad2deg( Real rad )
		{
			return 180.0f * rad / PI;
		}

		static Real map_to_minus_pi_to_pi( const Real a )
		{
			Real ret = a;
			while( ret > rec::math::PI )
			{
				ret -= 2.0f * rec::math::PI;
			}
			while( ret < -rec::math::PI )
			{
				ret += 2.0f * rec::math::PI;
			}
			return ret;
		}

		static bool realLess( const Real a, const Real b )
		{
			if( a < b - RealEpsilon )
			{
				return true;
			}

			return false;
		}

		static bool realGreater( const Real a, const Real b )
		{
			if( a > b + RealEpsilon )
			{
				return true;
			}

			return false;
		}

		static bool realEqual( const Real a, const Real b )
		{
			if( ( a >= 0 && b >= 0 ) || ( a < 0 && b < 0 ) )
			{
				if( fabs( a - b ) < RealEpsilon )
				{
					return true;
				}
			}
		
			return false;
		}
	}
}

#endif //_REC_MATH_UTILS_H_

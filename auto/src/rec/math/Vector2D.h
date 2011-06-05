#ifndef _REC_MATH_VECTOR2D_H_
#define _REC_MATH_VECTOR2D_H_

#include "rec/math/defines.h"
#include "rec/math/Vector.h"
#include <cmath>

namespace rec
{
	namespace math
	{
		typedef Vector< rec::math::Real, 2 > Vector2D;

		static Vector2D polarToVector2D( rec::math::Real orientation, rec::math::Real length )
		{
			return Vector2D( length * cos( orientation ), length * sin( orientation ) );
		}

		static void vector2DToPolar( const Vector2D& v, Real* orientation, Real* length )
		{
			*length = norm2( v );
			*orientation = atan2( v[1], v[0] );
		}

		static Vector2D rotate( const Vector2D& v, rec::math::Real phi )
		{
			Vector2D out;
			out[0] = v[0] * cos( phi ) - v[1] * sin( phi );
			out[1] = v[1] * cos( phi ) + v[0] * sin( phi );
			return out;
		}

		static Real cross( const Vector2D& a, const Vector2D& b )
		{
			return a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ];
		}
	}
}

#endif

#ifndef _REC_MATH_VECTOR3D_H_
#define _REC_MATH_VECTOR3D_H_

#include "rec/math/defines.h"
#include "rec/math/Vector.h"

namespace rec
{
	namespace math
	{
		typedef Vector< rec::math::Real, 3 > Vector3D;

		static Vector3D cross( const Vector3D& a, const Vector3D& b )
		{
			return a * b;
		}
	}
}

#endif

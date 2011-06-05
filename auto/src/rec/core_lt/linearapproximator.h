//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_LINEARAPPROXIMATOR_H_
#define _REC_CORE_LT_LINEARAPPROXIMATOR_H_

#include "rec/core_lt/PointF.h"

namespace rec
{
	namespace core_lt
	{
		template< typename InputIterator > double linearapproximator( InputIterator iter, InputIterator end, const double x )
		{
			double y = 0.0;

			if( iter != end )
			{
				if( x < (*iter).x() )
				{
					y = (*iter).y();
				}
				else
				{
					while( end != iter )
					{
						const rec::core_lt::PointF& p = (*iter);
						++iter;

						if( x >= p.x() )
						{
							if( end != iter )
							{
								if( x < (*iter).x() )
								{
									const rec::core_lt::PointF& p2 = (*iter);
									double dx = p2.x() - p.x();
									double dy = p2.y() - p.y();

									if( 0.0 == dx )
									{
										y = p.y();
									}
									else
									{
										double a = dy / dx;
										y = a * ( x - p.x() ) + p.y();
									}
								}
							}
							else
							{
								y = p.y();
							}
						}
					}
				}
			}

			return y;
		}

		inline double linearapproximator( const rec::core_lt::PointFVector& vec, const double x )
		{
			return linearapproximator( vec.begin(), vec.end(), x );
		}

		inline double linearapproximator( const rec::core_lt::PointF* pointArray, unsigned int size, const double x )
		{
			return linearapproximator( pointArray, pointArray + size, x );
		}
	}
}

#endif //_REC_CORE_LT_LINEARAPPROXIMATOR_H_

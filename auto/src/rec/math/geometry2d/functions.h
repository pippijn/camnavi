#ifndef _REC_MATH_GEOMETRY2D_FUNCTIONS_H_
#define _REC_MATH_GEOMETRY2D_FUNCTIONS_H_

#include "rec/math/Vector2D.h"
#include "rec/math/defines.h"

namespace rec
{
	namespace math
	{
		namespace geometry2d
		{
			/**
			Calculates intersection of two lines.
			The lines are each specified by a start point and a direction.
			*/
			static bool linelineIntersection( const Vector2D& lineStart1,
											  const Vector2D& lineDirection1,
											  const Vector2D& lineStart2,
											  const Vector2D& lineDirection2,
											  Vector2D* intersection = NULL )
			{
				if ( intersection )
				{
					( *intersection )[ 0 ] = 0;
					( *intersection )[ 1 ] = 0;
				}

				rec::math::Real test = lineDirection1[ 0 ] * lineDirection2[ 1 ] - lineDirection1[ 1 ] * lineDirection2[ 0 ];

				if ( test < 2 * RealEpsilon )
				{
					return false;
				}
				if ( intersection )
				{
					rec::math::Real f = ( lineStart2[ 0 ] * lineDirection2[ 1 ] - lineStart2[ 1 ] * lineDirection2[ 0 ] - lineStart1[ 0 ] * lineDirection2[ 1 ] + lineStart1[ 1 ] * lineDirection2[ 0 ] ) / test;
					*intersection = lineStart1 + f * lineDirection1;
				}
				return true;
			}

			/**
			Calculates intersection of a straight line and a circle.
			The line is specified by a start point and a direction.
			*/
			static int straightLineCircleIntersection( const Vector2D& lineStart,
													   const Vector2D& lineDirection,
													   const Vector2D& center,
													   rec::math::Real radius,
													   Vector2D* intersection1 = NULL,
													   Vector2D* intersection2 = NULL )
			{
				if ( intersection1 )
				{
					( *intersection1 )[ 0 ] = 0;
					( *intersection1 )[ 1 ] = 0;
				}
				if ( intersection2 )
				{
					( *intersection2 )[ 0 ] = 0;
					( *intersection2 )[ 1 ] = 0;
				}
				rec::math::Real a = lineDirection[ 0 ] * lineDirection[ 0 ] + lineDirection[ 1 ] * lineDirection[ 1 ];
				rec::math::Real b = 2 * ( lineStart[ 0 ] * lineDirection[ 0 ] - lineDirection[ 0 ] * center[ 0 ] + lineStart[ 1 ] * lineDirection[ 1 ] - lineDirection[ 1 ] * center[ 1 ] );

				rec::math::Real binom0 = ( lineStart[ 0 ] - center[ 0 ] );
				rec::math::Real binom1 = ( lineStart[ 1 ] - center[ 1 ] );
				rec::math::Real c = binom0 * binom0 + binom1 * binom1 - radius * radius;

				rec::math::Real test = b * b - 4 * a * c;

				if ( std::fabs( test ) < 2 * RealEpsilon )
				{
					rec::math::Real f = -b / ( 2 * a );
					rec::math::Vector2D intersection = lineStart + f * lineDirection;
					if ( intersection1 )
					{
						*intersection1 = intersection;
					}
					if ( intersection2 )
					{
						*intersection1 = intersection;
					}
					return 1;
				}
				if ( test < 0 )
				{
					return 0;
				}
				if ( intersection1 )
				{
					rec::math::Real f = ( -b - sqrt( test ) ) / ( 2 * a );
					*intersection1 = lineStart + f * lineDirection;
				}
				if ( intersection2 )
				{
					rec::math::Real f = ( -b + sqrt( test ) ) / ( 2 * a );
					*intersection2 = lineStart + f * lineDirection;
				}
				return 2;
			}

			/**
			Calculates intersection of a line and a circle. Optionally within the specified bounds.
			The line is specified by two points a and b.
			*/
			static int lineCircleIntersection( const Vector2D& a,
											   const Vector2D& b,
											   const Vector2D& center,
											   rec::math::Real radius,
											   bool withBounds = true,
											   Vector2D* intersection1 = NULL,
											   Vector2D* intersection2 = NULL )
			{
				if ( intersection1 )
				{
					( *intersection1 )[ 0 ] = 0;
					( *intersection1 )[ 1 ] = 0;
				}
				if ( intersection2 )
				{
					( *intersection2 )[ 0 ] = 0;
					( *intersection2 )[ 1 ] = 0;
				}
				Vector2D l_direction = normalize( b - a );
				Vector2D is1, is2;
				int num = straightLineCircleIntersection( a, l_direction, center, radius, &is1, &is2 );
				if ( num == 0 || withBounds == false )
				{
					if ( intersection1 )
					{
						*intersection1 = is1;
					}
					if ( intersection2 )
					{
						*intersection2 = is2;
					}
					return num;
				}
				int result = 0;
				if ( std::fabs( l_direction[ 0 ] ) < 2 * RealEpsilon )
				{
					if ( ( a[ 1 ] <= is1[ 1 ] && is1[ 1 ] <= b[ 1 ] ) ||
						 ( a[ 1 ] >= is1[ 1 ] && is1[ 1 ] >= b[ 1 ] ) )
					{
						result++;
					}
					if ( num > 1 )
					{
						if ( ( a[ 1 ] <= is2[ 1 ] && is2[ 1 ] <= b[ 1 ] ) ||
							 ( a[ 1 ] >= is2[ 1 ] && is2[ 1 ] >= b[ 1 ] ) )
						{
							if ( result == 0 )
								is1 = is2;
							result++;
						}
					}
				}
				else
				{
					if ( ( a[ 0 ] <= is1[ 0 ] && is1[ 0 ] <= b[ 0 ] ) ||
						 ( a[ 0 ] >= is1[ 0 ] && is1[ 0 ] >= b[ 0 ] ) )
					{
						result++;
					}
					if ( num > 1 )
					{
						if ( ( a[ 0 ] <= is2[ 0 ] && is2[ 0 ] <= b[ 0 ] ) ||
							 ( a[ 0 ] >= is2[ 0 ] && is2[ 0 ] >= b[ 0 ] ) )
						{
							if ( result == 0 )
								is1 = is2;
							result++;
						}
					}
				}
				if ( result == 2 )
				{
					if ( intersection1 )
						*intersection1 = is1;
					if ( intersection2 )
						*intersection2 = is2;
				}
				else if ( result == 1 )
				{
					if ( intersection1 )
						*intersection1 = is1;
					if ( intersection2 )
						*intersection2 = is1;
				}
				return result;
			}

			/**
			Projects a point on a line.
			The line is specified by a start point and a direction.
			*/
			static Vector2D projectPointOnLine( const Vector2D& lineStart, const Vector2D& lineDirection, const Vector2D& p )
			{
				Vector2D direction = normalize( lineDirection );
				rec::math::Vector2D sToP = p - lineStart;
				rec::math::Real factor = dot( direction, sToP ) / dot( direction, direction );
				direction *= factor;
				return lineStart + direction;
			}

			/**
			Calculates distance of a point from line formed by a and b
			*/
			static Real distanceFromLine( const Vector2D& a, const Vector2D& b, const Vector2D& point )
			{
				Vector2D v( rec::math::normalize( Vector2D( b - a ) ) );
				Vector2D v_perp( - v[ 1 ], v[ 0 ] );
				return rec::math::dot( (point - a), v_perp );
			}

			/**
			Calculates euclidian distance of two points
			*/
			static Real distance( const Vector2D& a, const Vector2D& b )
			{
				return static_cast< rec::math::Real >( rec::math::norm2( a - b ) );
			}

			// angle will be between -PI and +PI
			static float normAngle( float angle )
			{
				while( angle > (2*rec::math::PI) )
				{
					angle -= (2*rec::math::PI);
				}
				while( angle < 0 )
				{
					angle += (2*rec::math::PI);
				}
				if( angle > rec::math::PI )
				{
					angle = -(2*rec::math::PI) + angle;
				}
				return angle;
			}

			static float leftTurnDiff( float angleStart, float angleCur )
			{
				return normAngle(angleCur) - normAngle(angleStart);
			}

			static float rightTurnDiff( float angleStart, float angleCur )
			{
				return normAngle(angleStart) - normAngle(angleCur);
			}

			// returns the difference angle viewing direction of position p and point v
			static rec::math::Real diffAngle( const rec::math::Vector2D& v1, const rec::math::Vector2D& v2 )
			{
				rec::math::Vector2D diffV( v2 - v1 );
				float angle;
				if( fabs( diffV[0] ) < 0.0001f )
				{
					angle = rec::math::PI / 2.0f * (diffV[1] < 0.0f ? -1.0f : 1.0f);
				}
				else
				{
					angle = atan2( diffV[1], diffV[0] );
				}
				return normAngle( angle );
			}

			static rec::math::Vector2D orientedUnitVector( rec::math::Real angle )
			{
				return rec::math::Vector2D( (rec::math::Real)cos( angle ), (rec::math::Real)sin( angle ) );
			}

			// returns the (oriented) angle between AC and BC
			static rec::math::Real getAngle( const rec::math::Vector2D& a, const rec::math::Vector2D& b, const rec::math::Vector2D& c )
			{
				rec::math::Vector2D va = c - a;
				rec::math::Vector2D vb = c - b;
				rec::math::Real dp = rec::math::dot( va, vb );
				rec::math::Real cp = rec::math::cross( va, vb );
				return atan2( cp, dp );
			}
		}
	}
}

#endif

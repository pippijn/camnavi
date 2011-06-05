//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_POINTF_H_
#define _REC_CORE_LT_POINTF_H_

#include <cmath>
#include <cfloat>
#include <iostream>
#include "rec/core_lt/Vector.h"

namespace rec
{
	namespace core_lt
	{
		class PointF
		{
			friend std::istream& operator>>( std::istream& is, PointF& p );
		public:
			PointF()
				: _x( 0.0 )
				, _y( 0.0 )
			{
			}

			PointF(  double x,  double y )
				: _x( x )
				, _y( y )
			{
			}

			bool operator!=( const PointF& other ) const
			{
				return ( ( _x != other._x ) || ( _y != other._y ) );
			}

			bool operator==( const PointF& other ) const
			{
				return !operator!=( other );
			}

			PointF operator-( const PointF& other ) const
			{
				return PointF( _x - other._x, _y - other._y );
			}

			PointF operator+( const PointF& other ) const
			{
				return PointF( _x + other._x, _y + other._y );
			}

			double x() const { return _x; }
			void setX(  double x ) { _x = x; }

			double y() const { return _y; }
			void setY(  double y ) { _y = y; }

		private:
			double _x;
			double _y;
		};

		inline std::ostream& operator<<( std::ostream& os, const PointF& p )
		{
			os << "(";

			if( fabs( p.x() ) < FLT_EPSILON )
			{
				os << "0";
			}
			else
			{
				os << p.x();
			}
			os << " ";

			if( fabs( p.y() ) < FLT_EPSILON )
			{
				os << "0";
			}
			else
			{
				os << p.y();
			}
			
			os << ")";

			return os;
		}

		inline std::istream& operator>>( std::istream& is, PointF& p )
		{
			char ch;
			is >> ch;
			if( '(' != ch )
			{
				is.setstate( std::ios_base::failbit );
				return is;
			}

			if( is.good() )
			{
				is >> p._x;
			}
			else
			{
				p._x = 0;
			}

			if( is.good() )
			{
				is >> p._y;
			}
			else
			{
				p._y = 0;
			}

			if ( is.good() )
			{
				is >> ch; // closing bracket should be extracted from the stream
				// ch == ')'
			}
			return is;
		}
	
		typedef rec::core_lt::Vector< rec::core_lt::PointF > PointFVector;
	}
}

#endif //_REC_CORE_LT_POINTF_H_

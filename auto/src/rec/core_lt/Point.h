//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_POINT_H_
#define _REC_CORE_LT_POINT_H_

namespace rec
{
	namespace core_lt
	{
		class Point
		{
		public:
			Point()
				: _x( 0 )
				, _y( 0 )
			{
			}

			Point(  int x,  int y )
				: _x( x )
				, _y( y )
			{
			}

			bool operator!=( const Point& other ) const
			{
				return ( ( _x != other._x ) || ( _y != other._y ) );
			}

			bool operator==( const Point& other ) const
			{
				return !operator!=( other );
			}

			int x() const { return _x; }
			void setX(  int x ) { _x = x; }

			int y() const { return _y; }
			void setY(  int y ) { _y = y; }

		private:
			int _x;
			int _y;
		};
	}
}

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(rec::core_lt::Point)
#endif

#endif //_REC_CORE_LT_POINT_H_

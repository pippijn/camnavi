//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_LINE_H_
#define _REC_CORE_LT_LINE_H_

#include "rec/core_lt/Point.h"

#include <cmath>
#include <string>
#include <sstream>

namespace rec
{
	namespace core_lt
	{
		class Line
		{
		public:
			Line()
			{
			}

			Line( const Point& p0,  const Point& p1 )
				: _p0( p0 )
				, _p1( p1 )
			{
			}

			Line(  int x0,  int y0,  int x1,  int y1 )
				: _p0( x0, y0 )
				, _p1( x1, y1 )
			{
			}

			bool operator!=( const Line& other ) const
			{
				return ( ( _p0 != other._p0 ) || ( _p1 != other._p1 ) );
			}

			bool operator==( const Line& other ) const
			{
				return !operator!=( other );
			}

			const Point& p0() const { return _p0; }
			void setP0( const Point& p0 ) { _p0 = p0; }
			void setP0(  int x,  int y ) { _p0 = Point( x, y ); }

			const Point& p1() const { return _p1; }
			void setP1( const Point& p1 ) { _p1 = p1; }
			void setP1(  int x,  int y ) { _p1 = Point( x, y ); }

			int x0() const { return _p0.x(); }
			int y0() const { return _p0.y(); }

			int x1() const { return _p1.x(); }
			int y1() const { return _p1.y(); }

			double length() const
			{
				int dx = _p1.x() - _p0.x();
				int dy = _p1.y() - _p0.y();

				return sqrt( static_cast<double>( dx*dx ) + static_cast<double>( dy*dy ) );
			}

			double phi() const
			{
				int dx = _p1.x() - _p0.x();
				int dy = _p1.y() - _p0.y();

				return atan2( static_cast<double>( dy ), static_cast<double>( dx ) );
			}

			Point mid() const
			{
				int dx = _p1.x() - _p0.x();
				int dy = _p1.y() - _p0.y();

				Point mid( _p0.x() + dx / 2, _p0.y() + dy / 2 );
				return mid;
			}

		private:
			Point _p0;
			Point _p1;
		};

		static std::string toStdString( const Line& line )
		{
			std::ostringstream os;
			os << line.x0() << ";" << line.y0() << ";" << line.x1() << ";" << line.y1();
			return os.str();
		}

	}
}

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(rec::core_lt::Line)
#endif

#endif //_REC_CORE_LT_LINE_H_

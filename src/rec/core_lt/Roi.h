//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_ROI_H_
#define _REC_CORE_LT_ROI_H_

#include "rec/core_lt/Size.h"
#include "rec/core_lt/Point.h"

namespace rec
{
	namespace core_lt
	{
		class Roi
		{
		public:
			Roi()
			{
			}

			Roi( int x, int y, unsigned int width, unsigned int height )
				: _offset( x, y )
				, _size( width, height )
			{
			}

			bool isEmpty() const { return _size.isEmpty(); }

			bool operator!=( const Roi& other ) const
			{
				return ( ( _size != other._size ) || ( _offset != other._offset ) );
			}

			bool operator==( const Roi& other ) const
			{
				return !operator!=( other );
			}

			const Size& size() const { return _size; }
			void setSize( const Size& size ) { _size = size; }

			const Point& offset() const { return _offset; }
			void setOffset( const Point& offset ) { _offset = offset; }

		private:
			Point _offset;
			Size _size;
		};
	}
}

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(rec::core_lt::Roi)
#endif

#endif //_REC_CORE_LT_ROI_H_

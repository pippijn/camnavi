//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_SIZE_H_
#define _REC_CORE_LT_SIZE_H_

namespace rec
{
	namespace core_lt
	{
		class Size
		{
		public:
			Size()
				: _width( 0 )
				, _height( 0 )
			{
			}

			Size( unsigned int width, unsigned int height )
				: _width( width )
				, _height( height )
			{
			}

			bool isEmpty() const { return ( _width == 0 || _height == 0 ); }

			bool operator!=( const Size& other ) const
			{
				return ( ( _width != other._width ) || ( _height != other._height ) );
			}

			bool operator==( const Size& other ) const
			{
				return !operator!=( other );
			}

			unsigned int width() const { return _width; }
			void setWidth( unsigned int width ) { _width = width; }

			unsigned int height() const { return _height; }
			void setHeight( unsigned int height ) { _height = height; }

		private:
			unsigned int _width;
			unsigned int _height;
		};
	}
}

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(rec::core_lt::Size)
#endif

#endif //_REC_CORE_LT_SIZE_H_

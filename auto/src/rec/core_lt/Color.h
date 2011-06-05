//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_COLOR_H_
#define _REC_CORE_LT_COLOR_H_

#include "rec/core_lt/defines.h"

#include <boost/cstdint.hpp>

namespace rec
{
	namespace core_lt
	{
		class REC_CORE_LT_EXPORT Color
		{
		public:
			Color()
				: _argb( 0 )
			{
			}

			Color( boost::uint32_t argb )
				: _argb( argb )
			{
			}

			Color( boost::uint8_t r, boost::uint8_t g, boost::uint8_t b, boost::uint8_t a = 0 )
				: _argb( 0 )
			{
				setRgba( r, g, b, a );
			}

			boost::uint32_t argb() const { return _argb; }
			
			void setRgba( boost::uint8_t r, boost::uint8_t g, boost::uint8_t b, boost::uint8_t a = 0 )
			{
				setR( r );
				setG( g );
				setB( b );
				setA( a );
			}

			boost::uint8_t r() const { return static_cast<boost::uint8_t>( ( _argb & 0xFF0000 ) >> 16 ); }
			void setR( boost::uint8_t r ) { _argb &= 0xFF00FFFF; _argb |= ( r << 16 ); }

			boost::uint8_t g() const { return static_cast<boost::uint8_t>( ( _argb & 0xFF00 ) >> 8 ); }
			void setG( boost::uint8_t g ) { _argb &= 0xFFFF00FF; _argb |= ( g << 8 ); }
			
			boost::uint8_t b() const { return static_cast<boost::uint8_t>( _argb & 0xFF ); }
			void setB( boost::uint8_t b ) { _argb &= 0xFFFFFF00; _argb |= b; }

			boost::uint8_t a() const { return static_cast<boost::uint8_t>( ( _argb & 0xFF000000 ) >> 24 ); }
			void setA( boost::uint8_t a ) { _argb &= 0x00FFFFFF; _argb |= ( a << 24 ); }

			boost::uint8_t grey() const;

			bool operator==( const Color& other ) const;
			
			bool operator!=( const Color& other ) const;

			bool isNull() const { return ( 0 == _argb ); }

			static const Color null;

			static const Color white;
			static const Color black;
			
			static const Color transparent;

			static const Color red;
			static const Color darkred;
			static const Color lightred;

			static const Color green;
			static const Color darkgreen;
			static const Color lightgreen;
			
			static const Color blue;
			static const Color darkblue;
			static const Color lightblue;

		private:
			boost::uint32_t _argb;
		};
	}
}

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(rec::core_lt::Color)
#endif

#endif //_REC_CORE_LT_COLOR_H_

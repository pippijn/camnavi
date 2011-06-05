//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_UTILS_H_
#define _REC_CORE_LT_UTILS_H_

#include "rec/core_lt/defines.h"
#include "rec/core_lt/Exception.h"

#include <sstream>
#include <string>
#include <cmath>

namespace rec
{
	namespace core_lt
	{
		REC_CORE_LT_EXPORT void waitForKey();

    REC_CORE_LT_EXPORT void msleep( unsigned int ms );

		//// sleeps for the required time to meet a frequency of hz Hz
		REC_CORE_LT_EXPORT void msleep_HZ( unsigned int hz );

		template< typename T > std::string toString( T value, std::ios_base& ( *manipulator )( std::ios_base& ) = std::dec )
		{
			std::ostringstream os;
			os << manipulator << value;
			return os.str();
		}

		template< typename T > T fromString( const std::string& string, std::ios_base& ( *manipulator )( std::ios_base& ) = std::dec )
		{
			T value;
			std::istringstream is( string );
			if ( ( is >> manipulator >> value ).fail() )
				throw rec::core_lt::Exception( "Bad conversion" );
			return value;
		}

		static float mapToMinus180to180Degrees( float angle )
		{
			float angleOverflow = static_cast<float>( static_cast<int>( angle / 180.0f ) );

			if( angleOverflow > 0.0f )
			{
				angleOverflow = ceil( angleOverflow / 2.0f );
			}
			else
			{
				angleOverflow = floor( angleOverflow / 2.0f );
			}

			angle -= 360.0f * angleOverflow;

			return angle;
		}

		static float mapToMinusPItoPI( float angle )
		{
			float angleOverflow = static_cast<float>( static_cast<int>( angle / 3.141592653589793f ) );

			if( angleOverflow > 0.0f )
			{
				angleOverflow = ceil( angleOverflow / 2.0f );
			}
			else
			{
				angleOverflow = floor( angleOverflow / 2.0f );
			}

			angle -= 2 * 3.141592653589793f * angleOverflow;

			return angle;
		}
	}
}

#endif //_REC_CORE_LT_UTILS_H_

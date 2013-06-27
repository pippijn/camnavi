//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/Color.h"

using rec::core_lt::Color;

const Color Color::null( 0, 0, 0, 0 );

const Color Color::white( 255, 255, 255, 0 );
const Color Color::black( 0, 0, 0, 0 );

const Color Color::transparent( 0, 0, 0, 255 );

const Color Color::red( 255, 0, 0, 0 );
const Color Color::darkred( 100, 0, 0, 0 );
const Color Color::lightred( 255, 100, 100, 0 );

const Color Color::green( 0, 255, 0, 0 );
const Color Color::darkgreen( 0, 100, 0, 0 );
const Color Color::lightgreen( 100, 255, 100, 0 );

const Color Color::blue( 0, 0, 255, 0 );
const Color Color::darkblue( 0, 0, 100, 0 );
const Color Color::lightblue( 100, 100, 255, 0 );

boost::uint8_t Color::grey() const
{
	boost::uint32_t ret = r();
	ret += g();
	ret += b();

	ret /= 3;

	return static_cast<boost::uint8_t>( ret );
}

bool Color::operator==( const Color& other ) const
{
	if( other._argb == _argb )
	{
		return true;
	}

	return false;
}

bool Color::operator!=( const Color& other ) const
{
	return !operator==( other );
}

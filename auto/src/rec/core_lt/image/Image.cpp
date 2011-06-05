//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/core_lt/image/Image.h"
#include "rec/core_lt/image/ImageImpl.h"
#include "rec/core_lt/Log.h"

using namespace rec::core_lt::image;

Image::Image()
: _impl( new ImageImpl )
{
}

Image::~Image()
{
	delete _impl;
}

Image::Image( const ImageInfo& info )
:  _impl( new ImageImpl( info ) )
{
}

Image::Image( const Image& other )
: _impl( new ImageImpl( *other._impl ) )
{
}

Image& Image::operator=( const Image& other )
{
	delete _impl;
	_impl = new ImageImpl( *other._impl );
	return *this;
}

bool Image::isNull() const
{
	return _impl->data.isNull();
}

unsigned char* Image::data()
{
	return _impl->data.data();
}

char* Image::data_s()
{
	return _impl->data.data_s();
}

const unsigned char* Image::constData() const
{
	return _impl->data.constData();
}

const char* Image::constData_s() const
{ 
	return _impl->data.constData_s();
}

rec::core_lt::memory::ByteArray Image::toByteArray() const
{ 
	return _impl->data;
}

const ImageInfo& Image::info() const
{
	return _impl->info;
}

unsigned int Image::step() const
{
	return _impl->step;
}

rec::core_lt::Color Image::pixel( const rec::core_lt::Point& point ) const
{
	return pixel( point.x(), point.y() );
}

rec::core_lt::Color Image::pixel( unsigned int x, unsigned int y ) const
{
	if( x >= info().width )
	{
		return rec::core_lt::Color();
	}
	if( y >= info().height )
	{
		return rec::core_lt::Color();
	}

	unsigned int byte = y * step() + x * info().numChannels;
	assert( byte < _impl->data.size() );

	const unsigned char* p = _impl->data.constData() + byte;

	if( 3 == info().numChannels )
	{
		return rec::core_lt::Color( p[0], p[1], p[2] );  
	}
	else if( 1 == info().numChannels )
	{
		return rec::core_lt::Color( p[0], p[0], p[0] );  
	}
	else
	{
		return rec::core_lt::Color();
	}
}

unsigned int rec::core_lt::image::getStepSize( unsigned int width, unsigned int channels )
{
	//adjust here to different PixelType sizes
	unsigned int step = width * channels;
	if( (step & 0x1F) != 0 )
	{
		// align rows to 32byte boundary
		step = (step & (~0x1F)) + 0x20;
	}
	return step;
}


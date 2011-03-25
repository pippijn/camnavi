//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/image/conv.h"
#include "rec/core_lt/image/jpeg.hpp"
#include "rec/core_lt/image/bmp.hpp"
#include "rec/core_lt/memory/DataStream.h"
#include "rec/core_lt/memory/FileIO.h"

#include <cstring>
#include <boost/cstdint.hpp>

rec::core_lt::image::Image fromJpeg( const rec::core_lt::memory::ByteArrayConst& data )
{
	if( data.isEmpty() )
	{
		return rec::core_lt::image::Image();
	}

	return rec::core_lt::image::jpg_decompress( data );
}

rec::core_lt::image::Image fromBitmap( const rec::core_lt::memory::ByteArrayConst& data )
{
	return rec::core_lt::image::bmp_load( data );
}

rec::core_lt::image::Image rec::core_lt::image::loadFromData( const rec::core_lt::memory::ByteArrayConst& data, const std::string& format )
{
	if( format.empty() )
	{
		rec::core_lt::memory::DataStream is( data, rec::core_lt::memory::DataStream::LittleEndian );

		boost::uint16_t imgStart;
		is >> imgStart;

		if( !is.isOk() )
		{
			return rec::core_lt::image::Image();
		}
		
		//jpg is BigEndian encoded so test for d8ff and not ffd8
		if( 0xd8ff == imgStart )
		{
			return fromJpeg( data );
		}
		
		if( 19778 == imgStart )
		{
			return fromBitmap( data );
		}
	}
	else if( "bmp" == format )
	{
		return fromBitmap( data );
	}
	else if( "jpg" == format || "jpeg" == format )
	{
		return fromJpeg( data );
	}

	return rec::core_lt::image::Image();
}

bool rec::core_lt::image::jpegImageWidthAndHeight(
				const rec::core_lt::memory::ByteArrayConst& data,
				unsigned int* imageWidth,
				unsigned int* imageHeight )
{
	rec::core_lt::image::ImageInfo info = rec::core_lt::image::jpg_info( data );

	if( info.isNull() )
	{
		return false;
	}
	else
	{
		*imageWidth = info.width;
		*imageHeight = info.height;
		return true;
	}
}

bool rec::core_lt::image::jpegImageWidthAndHeight(
				const unsigned char* data,
				unsigned int dataSize,
				unsigned int* imageWidth,
				unsigned int* imageHeight )
{
	rec::core_lt::memory::ByteArrayConst ba = rec::core_lt::memory::ByteArrayConst::fromRawData( data, dataSize );

	return jpegImageWidthAndHeight( ba, imageWidth, imageHeight );
}

bool rec::core_lt::image::loadImageFromData(
				const unsigned char* data,
				unsigned int dataSize,
				unsigned char* imageBuffer,
				unsigned int imageBufferSize,
				unsigned int* imageWidth,
				unsigned int* imageHeight )
{
	rec::core_lt::memory::ByteArrayConst ba = rec::core_lt::memory::ByteArrayConst::fromRawData( data, dataSize );

	rec::core_lt::image::Image image = rec::core_lt::image::loadFromData( ba );

	if( image.isNull() )
	{
		return false;
	}
	else
	{
		if( image.toByteArray().size() > imageBufferSize )
		{
			return false;
		}

		memcpy( (void*)imageBuffer, (const void*)image.toByteArray().constData(), image.toByteArray().size() );
		*imageWidth = image.info().width;
		*imageHeight = image.info().height;

		return true;
	}
}

rec::core_lt::image::Image rec::core_lt::image::loadFromFile( const std::string& filename, const std::string& format )
{
	rec::core_lt::memory::ByteArray data = rec::core_lt::memory::read( filename );
	return loadFromData( data, format );
}

rec::core_lt::memory::ByteArray rec::core_lt::image::toJpeg( const rec::core_lt::image::Image& image )
{
	return rec::core_lt::image::jpg_compress( image );
}

rec::core_lt::memory::ByteArray rec::core_lt::image::toBitmap( const rec::core_lt::image::Image& image )
{
	return rec::core_lt::image::bmp_save( image );
}

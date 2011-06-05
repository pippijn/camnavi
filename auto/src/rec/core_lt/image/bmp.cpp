//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/image/bmp.hpp"
#include "rec/core_lt/memory/DataStream.h"

#include <cstring>
#include <boost/cstdint.hpp>

#ifdef WIN32
#pragma pack(1)
#define PACK_ATTRIB
#else
#define PACK_ATTRIB __attribute__ ((packed))
#endif

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			struct bmp_header
			{
				// BITMAPFILEHEADER
				boost::uint16_t bfType;           // =19778  must always be set to 'BM' to declare that this is a .bmp-file.
				boost::uint32_t bfSize;           // specifies the size of the file in bytes
				boost::uint16_t bfReserved1;      // must always be set to zero
				boost::uint16_t bfReserved2;      // must always be set to zero
				boost::uint32_t bfOffBits;        // specifies the offset from the beginning of the file to the bitmap data
				// BITMAPINFOHEADER
				boost::uint32_t biSize;           // =40 specifies the size of the BITMAPINFOHEADER structure, in bytes
				boost::uint32_t biWidth;          // specifies the width of the image, in pixels
				boost::uint32_t biHeight;         // specifies the height of the image, in pixels
				boost::uint16_t biPlanes;         // =1 specifies the number of planes of the target device
				boost::uint16_t biBitCount;       // =8 specifies the number of bits per pixel
				boost::uint32_t biCompression;    // Specifies the type of compression, usually set to zero (no compression)
				boost::uint32_t biImgSize;           // Specifies size of image data if there is no compression
				boost::uint32_t biXPelsPerMeter;  // specifies the the horizontal pixels per meter, usually set to zero
				boost::uint32_t biYPelsPerMeter;  // specifies the the vertical pixels per meter, usually set to zero
				boost::uint32_t biClrUsed;        // # of colors in the bitmap, if zero the # of colors is calced using biBitCount
				boost::uint32_t biClrImportant;   // # of colors that are 'important', if set to zero, all colors are important
			} PACK_ATTRIB;

			struct pal_entry
			{
				boost::uint8_t b;
				boost::uint8_t g;
				boost::uint8_t r;
				boost::uint8_t rsvd;
			} PACK_ATTRIB;
		}
	}
}

#ifdef WIN32
#pragma pack()
#endif

using namespace rec::core_lt::image;

unsigned int rec::core_lt::image::bmp_rowsize( unsigned int width, unsigned int numChannels, unsigned int* numFillBytes )
{
	float unaligned = static_cast<float>( ( numChannels * 8 * width ) ) / 32;
	float top = ceil( unaligned );

	if( numFillBytes )
	{
		*numFillBytes = static_cast< unsigned int >( ( top - unaligned ) * 4 );
	}
	unsigned int rowsize = 4 * static_cast<unsigned int>( top );
	return rowsize;
}

rec::core_lt::image::Image rec::core_lt::image::bmp_load( const rec::core_lt::memory::ByteArrayConst& data )
{
	struct bmp_header bmp_hdr;

	if( data.size() < sizeof(bmp_hdr) )
	{
		return rec::core_lt::image::Image();
	}

	rec::core_lt::memory::DataStream is( data, rec::core_lt::memory::DataStream::LittleEndian );

	boost::uint16_t u16;
	boost::uint32_t u32;

	is >> u16;
	bmp_hdr.bfType = u16;

	is >> u32;
	bmp_hdr.bfSize = u32;

	is >> u16;
	bmp_hdr.bfReserved1 = u16;

	is >> u16;
	bmp_hdr.bfReserved2 = u16;

	is >> u32;
	bmp_hdr.bfOffBits = u32;

	is >> u32;
	bmp_hdr.biSize = u32;

	is >> u32;
	bmp_hdr.biWidth = u32;

	is >> u32;
	bmp_hdr.biHeight = u32;

	is >> u16;
	bmp_hdr.biPlanes = u16;

	is >> u16;
	bmp_hdr.biBitCount = u16;

	is >> u32;
	bmp_hdr.biCompression = u32;

	is >> u32;
	bmp_hdr.biImgSize = u32;

	is >> u32;
	bmp_hdr.biXPelsPerMeter = u32;

	is >> u32;
	bmp_hdr.biYPelsPerMeter = u32;

	is >> u32;
	bmp_hdr.biClrUsed = u32;

	is >> u32;
	bmp_hdr.biClrImportant = u32;

	if( bmp_hdr.bfType != 19778 )
	{
		return rec::core_lt::image::Image();
	}

	if( 24 == bmp_hdr.biBitCount )
	{
		if( bmp_hdr.bfOffBits + bmp_hdr.biWidth * bmp_hdr.biHeight * 3 > data.size() )
		{
			return rec::core_lt::image::Image();
		}

		rec::core_lt::image::ImageInfo info( bmp_hdr.biWidth, bmp_hdr.biHeight );
		rec::core_lt::image::Image img( info );

		unsigned int numFillBytes;
		unsigned int bmprowsize = bmp_rowsize( img.info().width, img.info().numChannels, &numFillBytes );

		unsigned int bmpRow = img.info().height - 1;
		for( unsigned int imgRow=0; imgRow<img.info().height; ++imgRow )
		{
			unsigned char* imgData = img.data() + imgRow * img.step();
			const unsigned char* bitmapData = data.constData() + bmp_hdr.bfOffBits + bmpRow * bmprowsize;

			for( unsigned int x=0; x<img.info().width; ++x )
			{
				imgData[ 2 ] = bitmapData[ 0 ];
				imgData[ 1 ] = bitmapData[ 1 ];
				imgData[ 0 ] = bitmapData[ 2 ];

				imgData += 3;
				bitmapData += 3;
			}

			--bmpRow;
		}

		return img;
	}
	else
	{
	}

	return rec::core_lt::image::Image();
}

unsigned int rec::core_lt::image::bmp_calculate_header_size( const rec::core_lt::image::Image& image )
{
	return sizeof(bmp_header) + (image.info().numChannels == 3 ? 0 : 256*sizeof(pal_entry) );
}

unsigned int rec::core_lt::image::bmp_calculate_size( const rec::core_lt::image::Image& image )
{
	return bmp_calculate_header_size( image ) + bmp_rowsize( image.info().width, image.info().numChannels ) * image.info().height;
}

unsigned int rec::core_lt::image::bmp_create_header( unsigned char* pData, unsigned int bufferSize, const rec::core_lt::image::Image& image )
{
	struct bmp_header bmp_hdr;
	struct pal_entry pal;
	bool color = ( image.info().numChannels == 3 );

	bmp_hdr.bfType = 19778;
	bmp_hdr.bfSize = bmp_calculate_size( image );
	bmp_hdr.bfReserved1 = 0;
	bmp_hdr.bfReserved2 = 0;
	bmp_hdr.bfOffBits = sizeof(bmp_hdr) + (color?0:256*sizeof(pal));
	bmp_hdr.biSize = 40;
	bmp_hdr.biWidth = image.info().width;
	bmp_hdr.biHeight = image.info().height;
	bmp_hdr.biPlanes = 1;
	bmp_hdr.biBitCount = color?24:8;
	bmp_hdr.biCompression = 0;
	bmp_hdr.biImgSize = 0;//bmp_data_size( image.info().width, image.info().height, image.info().numChannels );
	bmp_hdr.biXPelsPerMeter = 0;
	bmp_hdr.biYPelsPerMeter = 0;
	bmp_hdr.biClrUsed = 0;
	bmp_hdr.biClrImportant = 0;

	memcpy( (void*) pData, (const void*) &bmp_hdr, sizeof( bmp_hdr ) );
	pData += sizeof( bmp_hdr );
	if( !color )
	{
		for( int i = 0; i < 256; i++)
		{
			pal.b = pal.g = pal.r = i;
			pal.rsvd = 0;
			memcpy( (void*) pData, (const void*) &pal, sizeof( pal ) );
			pData += sizeof( pal );
		}
	}

	return bmp_calculate_header_size( image );
}

rec::core_lt::memory::ByteArray rec::core_lt::image::bmp_save( const rec::core_lt::image::Image& image )
{
	int i;
	int j;
	bool color = ( image.info().numChannels == 3 );

	unsigned int size = bmp_calculate_size( image );
	rec::core_lt::memory::ByteArray buffer( size );

	unsigned int hs = bmp_create_header( buffer.data(), buffer.size(), image );

	unsigned int rowsize = bmp_rowsize( image.info().width, image.info().numChannels );

	for( i = image.info().height - 1; i >= 0; i-- )
	{
		const unsigned char* imgData = image.constData() + i * image.step();
		unsigned char* pData = buffer.data() + hs + ( image.info().height - i - 1 ) * rowsize;
		if( color )
		{
			// switch rgb to bgr
			for( j = 0; j < (int) image.info().width; ++j )
			{
				pData[ 0 ] = imgData[ 2 ];
				pData[ 1 ] = imgData[ 1 ];
				pData[ 2 ] = imgData[ 0 ];
				pData += 3;
				imgData += 3;
			}
		}
		else
		{
			memcpy( static_cast< void* >( pData ), static_cast< const void* >( imgData ), image.info().width );
			pData += image.info().width;
		}
	}
	return buffer;
}

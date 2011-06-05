//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

/*this software is based in part on the work of
the Independent JPEG Group*/

#ifndef _REC_CORE_LT_IMAGE_CONV_H_
#define _REC_CORE_LT_IMAGE_CONV_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/image/Image.h"
#include "rec/core_lt/memory/ByteArrayConst.h"
#include "rec/core_lt/memory/ByteArray.h"

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			///@param format Leave empty to guess the image format
			///              Supported format: "bmp", "jpg", "jpeg"
			REC_CORE_LT_EXPORT Image loadFromData( const rec::core_lt::memory::ByteArrayConst& data, const std::string& format = std::string() );
			
			REC_CORE_LT_EXPORT Image loadFromFile( const std::string& filename, const std::string& format = std::string() );

			REC_CORE_LT_EXPORT rec::core_lt::memory::ByteArray toBitmap( const Image& image );

			REC_CORE_LT_EXPORT rec::core_lt::memory::ByteArray toJpeg( const Image& image );
			
			REC_CORE_LT_EXPORT rec::core_lt::memory::ByteArray toPng( const Image& image );

			REC_CORE_LT_EXPORT bool jpegImageWidthAndHeight(
				const rec::core_lt::memory::ByteArrayConst& data,
				unsigned int* imageWidth,
				unsigned int* imageHeight );

			/**
			Read width and height from jpg image data.
			@param imageWidth Stores the width of the uncompressed image.
			@param imageHeight Stores the height of the uncompressed image.
			@return Returns TRUE (1) if image is a valid jpeg image. Returns FALSE (0) otherwise.
			*/
			REC_CORE_LT_EXPORT bool jpegImageWidthAndHeight(
				const unsigned char* data,
				unsigned int dataSize,
				unsigned int* imageWidth,
				unsigned int* imageHeight );

			/**
			Loads image from data.
			@param data Image data (Bitmap or JPEG images).
			@param dataSize Size of data.
			@param imageBuffer The uncompressed image is copied to this buffer. The function fails if imageBufferSize is smaller than the uncompressed image.
			The image is stored as RGB interleaved image with three channels and one byte per channel. Image step is equal to image width.
			@param imageBufferSize Size of imageBuffer.
			@param imageWidth Stores the width of the uncompressed image.
			@param imageHeight Stores the height of the uncompressed image.
			@return Returns TRUE (1) if image was successfully loaded. Returns FALSE (0) otherwise.
			*/
			REC_CORE_LT_EXPORT bool loadImageFromData(
				const unsigned char* data,
				unsigned int dataSize,
				unsigned char* imageBuffer,
				unsigned int imageBufferSize,
				unsigned int* imageWidth,
				unsigned int* imageHeight );
		}
	}
}

#endif //_REC_CORE_LT_IMAGE_CONV_H_



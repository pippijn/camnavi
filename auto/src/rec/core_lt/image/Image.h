//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_IMAGE_IMAGE_H_
#define _REC_CORE_LT_IMAGE_IMAGE_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/memory/ByteArray.h"

#include "rec/core_lt/image/ImageInfo.h"

#include "rec/core_lt/Map.h"
#include "rec/core_lt/Roi.h"
#include "rec/core_lt/Line.h"
#include "rec/core_lt/Vector.h"
#include "rec/core_lt/image/Palette.h"
#include "rec/core_lt/image/Segment.h"

namespace rec
{
	namespace core_lt
	{
		namespace variant
		{
			class Variant;
		}
	}
}

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			class ImageImpl;

			class REC_CORE_LT_EXPORT Image
			{
			public:
				
				/// Constructs empty image with Format_Undefined
				Image();

				Image( const Image& other );

				Image& operator=( const Image& other );


				~Image();

				Image( const ImageInfo& info );

				bool isNull() const;

				unsigned char* data();

				char* data_s();

				const unsigned char* constData() const;

				const char* constData_s() const;

				rec::core_lt::memory::ByteArray toByteArray() const;

				const ImageInfo& info() const;

				/**
				* step is number of bytes from 1st byte in line a to 1st byte in line a+1
				*/
				unsigned int step() const;

				rec::core_lt::Color pixel( const rec::core_lt::Point& point ) const;

				rec::core_lt::Color pixel( unsigned int x, unsigned int y ) const;

			private:
				ImageImpl* _impl;
			};

			REC_CORE_LT_EXPORT unsigned int getStepSize( unsigned int width, unsigned int channels );
		}
	}
}

#endif

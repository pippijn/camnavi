//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_IMAGE_IMAGEIMPL_H_
#define _REC_CORE_LT_IMAGE_IMAGEIMPL_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/image/ImageInfo.h"

#include "rec/core_lt/memory/ByteArray.h"

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			class ImageImpl
			{
			public:
				ImageImpl()
					: step( 0 )
				{
				}

				ImageImpl( const ImageInfo& info_ )
					: info( info_ )
					, step( 0 )
				{
					if( !info.isNull() )
					{
						step = getStepSize( info.width, info.numChannels );
						data = rec::core_lt::memory::ByteArray( step * info.height );
					}
				}

				ImageInfo info;
				unsigned int step;
				rec::core_lt::memory::ByteArray data;
			};
		}
	}
}

#endif //_REC_CORE_LT_IMAGE_IMAGEIMPL_H_

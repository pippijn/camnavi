//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_IMAGERECEIVEDEVENT_H_
#define _REC_ROBOTINO_COM_IMAGERECEIVEDEVENT_H_

#include <QByteArray>

#include "rec/robotino/com/events/ComEvent.h"
#include "rec/robotino/com/ComImpl.hh"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class ImageReceivedEvent : public ComEvent
			{
			public:
				ImageReceivedEvent( const QByteArray& data_,
					ComImpl::ImageType_t type_,
					unsigned int width_,
					unsigned int height_,
					unsigned int numChannels_,
					unsigned int bitsPerChannel_,
					unsigned int step_ )
					: ComEvent( ComEvent::ImageReceivedEventId )
					, data( data_ )
					, type( type_ )
					, width( width_ )
					, height( height_ )
					, numChannels( numChannels_ )
					, bitsPerChannel( bitsPerChannel_ )
					, step( step_ )
				{
				}

				const QByteArray data;
				const ComImpl::ImageType_t type;
				const unsigned int width;
				const unsigned int height;
				const unsigned int numChannels;
				const unsigned int bitsPerChannel;
				const unsigned int step;
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_IMAGERECEIVEDEVENT_H_

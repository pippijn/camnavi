//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_CONNECTIONCLOSEDEVENT_H_
#define _REC_ROBOTINO_COM_CONNECTIONCLOSEDEVENT_H_

#include "rec/robotino/com/events/ComEvent.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class ConnectionClosedEvent : public ComEvent
			{
			public:
				ConnectionClosedEvent()
					: ComEvent( ComEvent::ConnectionClosedEventId )
				{
				}
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_CONNECTIONCLOSEDEVENT_H_

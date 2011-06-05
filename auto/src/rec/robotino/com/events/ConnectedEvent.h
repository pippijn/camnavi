//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_CONNECTEDEVENT_H_
#define _REC_ROBOTINO_COM_CONNECTEDEVENT_H_

#include "rec/robotino/com/events/ComEvent.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class ConnectedEvent : public ComEvent
			{
			public:
				ConnectedEvent()
					: ComEvent( ComEvent::ConnectedEventId )
				{
				}
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_CONNECTEDEVENT_H_

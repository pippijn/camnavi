//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_INFORECEIVEDEVENT_H_
#define _REC_ROBOTINO_COM_INFORECEIVEDEVENT_H_

#include "rec/robotino/com/events/ComEvent.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class InfoReceivedEvent : public ComEvent
			{
			public:
				InfoReceivedEvent()
					: ComEvent( ComEvent::InfoReceivedEventId )
				{
				}
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_INFORECEIVEDEVENT_H_

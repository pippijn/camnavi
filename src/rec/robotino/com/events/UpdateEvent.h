//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_UPDATEEVENT_H_
#define _REC_ROBOTINO_COM_UPDATEEVENT_H_

#include "rec/robotino/com/events/ComEvent.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class UpdateEvent : public ComEvent
			{
			public:
				UpdateEvent()
					: ComEvent( ComEvent::UpdateEventId )
				{
				}
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_UPDATEEVENT_H_

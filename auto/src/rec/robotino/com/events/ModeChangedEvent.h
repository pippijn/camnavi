//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_MODECHANGEDEVENT_H_
#define _REC_ROBOTINO_COM_MODECHANGEDEVENT_H_

#include "rec/robotino/com/events/ComEvent.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class ModeChangedEvent : public ComEvent
			{
			public:
				ModeChangedEvent( bool isPassiveMode_ )
					: ComEvent( ComEvent::ModeChangedEventId )
					, isPassiveMode( isPassiveMode_ )
				{
				}

				const bool isPassiveMode;
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_MODECHANGEDEVENT_H_

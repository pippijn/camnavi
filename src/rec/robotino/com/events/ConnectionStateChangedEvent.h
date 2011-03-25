//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_CONNECTIONSTATECHANGEDEVENT_H_
#define _REC_ROBOTINO_COM_CONNECTIONSTATECHANGEDEVENT_H_

#include "rec/robotino/com/events/ComEvent.h"
#include "rec/robotino/com/Com.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class ConnectionStateChangedEvent : public ComEvent
			{
			public:
				ConnectionStateChangedEvent( Com::ConnectionState newState_, Com::ConnectionState oldState_ )
					: ComEvent( ComEvent::ConnectionStateChangedEventId )
					, newState( newState_ )
					, oldState( oldState_ )
				{
				}

				const Com::ConnectionState newState;
				const Com::ConnectionState oldState;
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_CONNECTIONSTATECHANGEDEVENT_H_

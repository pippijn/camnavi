//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_ERROREVENT_H_
#define _REC_ROBOTINO_COM_ERROREVENT_H_

#include "rec/robotino/com/events/ComEvent.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class ErrorEvent : public ComEvent
			{
			public:
				ErrorEvent( Com::Error error_, const std::string& errorStr_ )
					: ComEvent( ComEvent::ErrorEventId )
					, error( error_ )
					, errorStr( errorStr_ )
				{
				}

				const Com::Error error;
				const std::string errorStr;
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_ERROREVENT_H_

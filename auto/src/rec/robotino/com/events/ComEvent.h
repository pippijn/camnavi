//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_COMEVENT_H_
#define _REC_ROBOTINO_COM_COMEVENT_H_

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class ComEvent
			{
			public:
				typedef enum { ErrorEventId,
					ConnectedEventId,
					ConnectionClosedEventId,
					ConnectionStateChangedEventId,
					UpdateEventId,
					ModeChangedEventId,
					ImageReceivedEventId,
					InfoReceivedEventId,
				} Id_t;

				ComEvent( Id_t id_ )
					: id( id_ )
				{
				}

				virtual ~ComEvent()
				{
				}

				const Id_t id; 
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_COMEVENT_H_

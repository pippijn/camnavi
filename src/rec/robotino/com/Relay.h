//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_RELAY_H_
#define _REC_ROBOTINO_COM_RELAY_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{     
		namespace com
		{
			/**
			* @brief	Represents  a relay.
			*/
			class
#ifdef WIN32
#  ifdef rec_robotino_com_EXPORTS
		__declspec(dllexport)
#endif
#  ifdef rec_robotino_com2_EXPORTS
		__declspec(dllexport)
#endif
#  ifdef rec_robotino_com3_EXPORTS
		__declspec(dllexport)
#endif
#endif
			Relay : public Actor
			{
			public:
				Relay();

				/**
				* @return Returns the number of relays
				*/
				static unsigned int numRelays();

				/**
				* Sets the relay number.
				*
				* @param n	The relay number. Range: [0; Robotstate::numRelays]
				* @throws	RobotinoException if relay number is invalid.
				*/
				void setRelayNumber( unsigned int n );

				/**
				* Sets the relay to the given value.
				*
				* @param on	The value the relay will be set to.
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				void setValue( bool on );

			private:
				unsigned int _relayNumber;
			};
		}
	}
}
#endif

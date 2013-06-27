//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_BUMPER_H_
#define _REC_ROBOTINO_COM_BUMPER_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{    
		namespace com
		{ 
			/**
			* @brief	Represents a bumper.
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
			Bumper : public Actor
			{
			public:
				/**
				* Returns the current state of the bumper.
				* @return	TRUE if the bumper has contact, FALSE otherwise
				* @throws	RobotinoException if the underlying communication object is invalid
				* @see Actor::setComId
				*/
				bool value() const;
			};
		}
	}
}
#endif

//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_POWERMANAGEMENT_H_
#define _REC_ROBOTINO_COM_POWERMANAGEMENT_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{   
		namespace com
		{

			/**
			* @brief	Represents the power management of Robotino.
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
			PowerManagement : public Actor
			{
			public:
				/**
				* Retrieves the current power drain.
				*
				* @return	The power drain in mA.
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				float current() const;

				/**
				* Retrieves the battery voltage.
				*
				* @return	Battery voltage in V.
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				float voltage() const;
			};
		}
	}
}
#endif


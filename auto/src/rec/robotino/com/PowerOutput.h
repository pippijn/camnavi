//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_POWEROUTPUT_H_
#define _REC_ROBOTINO_COM_POWEROUTPUT_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	Represents a digital output device.
			*
			* Caution: You must not use the PowerOutput class to drive Robotino's gripper. The gripper will
			* be physically destroyed if current is not limited. Use the Gripper class instead.
			* If there is at least one Gripper object assigend to the same Com object as this
			* PowerOutput, the PowerOutput is disabled.
			@see Gripper
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
			PowerOutput : public Actor
			{
			public:
				PowerOutput();

				/**
				* Sets the current set point of the power output.
				*
				* @param controlPoint	The control point. Range from -100 to 100.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				void setValue( float controlPoint );

				/**
				* @return The current delivered by the power output in A.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				float current() const;

				/**
				* The current is measured by a 10 bit adc and is not converted into A.
				* @return The current delivered by the power output. Range from 0 to 1023.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				unsigned short rawCurrentMeasurment() const;
			};
		}
	}
}
#endif //_REC_ROBOTINO_COM_POWEROUTPUT_H_


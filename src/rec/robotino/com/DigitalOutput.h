//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_DIGITALOUTPUT_H_
#define _REC_ROBOTINO_COM_DIGITALOUTPUT_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	Represents a digital output device.
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
			DigitalOutput : public Actor
			{
			public:
				DigitalOutput();

				/**
				* @return Returns the number of digital outputs
				* @throws nothing
				*/
				static unsigned int numDigitalOutputs();

				/**
				* Sets the number of this digital output device.
				*
				* @param n	The output number. Range [0; Robotstate::numDigitalOutputs]
				* @throws	RobotinoException if the given output number is invalid.
				*/
				void setOutputNumber( unsigned int n );

				/**
				* Sets the current value of the specified output device.
				*
				* @param on	The output value of this device.
				* @throws	RobotinoException if the underlying communication object is invalid
				* @see		setOutputNumber(), Actor::setComId
				*/
				void setValue( bool on );

			private:
				unsigned int _outputNumber;
			};
		}
	}
}
#endif

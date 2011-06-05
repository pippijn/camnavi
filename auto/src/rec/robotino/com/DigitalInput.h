//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_DIGITALINPUT_H_
#define _REC_ROBOTINO_COM_DIGITALINPUT_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	Represents a digital input device.
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
			DigitalInput : public Actor
			{
			public:
				DigitalInput();

				/**
				* @return Returns the number of digital inputs
				* @throws nothing
				*/
				static unsigned int numDigitalInputs();

				/**
				* Sets the number of this digital input device.
				*
				* @param n	The input number. Range [0; Robotstate::numDigitalInputs]
				* @throws	RobotinoException if the given input number is invalid.
				*/
				void setInputNumber( unsigned int n );

				/**
				* Returns the current value of the specified input device.
				*
				* @return	The current value of the specified digital input
				* @throws	RobotinoException if the underlying communication object is invalid
				* @see		setInputNumber(), Actor::setComId
				*/
				bool value() const;

			private:
				unsigned int _inputNumber;
			};
		}
	}
}
#endif

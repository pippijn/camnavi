//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_ANALOGINPUT_H_
#define _REC_ROBOTINO_COM_ANALOGINPUT_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	An analog Input.
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
			AnalogInput : public Actor
			{
			public:
				AnalogInput();

				/**
				* @return Returns the number of analog inputs.
				*/
				static unsigned int numAnalogInputs();

				/**
				* Sets the number of this analog input device.
				*
				* @param n	The input number. Range [0; Robotstate::numAnalogInputs]
				* @throws	RobotinoException if the given input number is out of range.
				*/
				void setInputNumber( unsigned int n );

				/**
				* Returns the current value of the specified input device.
				*
				* @return	The current value of the specified analog input
				* @throws	Exception if the underlying communication object is invalid
				* @see		setInputNumber(), Actor::setComId
				*/
				float value() const;

			private:
				unsigned int _inputNumber;
			};
		}
	}
}

#endif

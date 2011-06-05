//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_MOTOR_H_
#define _REC_ROBOTINO_COM_MOTOR_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	Represents a single motor
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
			Motor : public Actor
			{
			public:
				Motor();

				virtual ~Motor();

				/**
				* @return Returns the number of drive motors on Robotino
				* @throws nothing
				*/
				static unsigned int numMotors();

				/**
				* Sets the number of this motor.
				*
				* @param number	number of this motor
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				void setMotorNumber( unsigned int number );

				/**
				* Sets the setpoint speed of this motor.
				*
				* @param speed	Set point speed in rpm.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				void setSpeedSetPoint( float speed );

				/**
				* Resets the position of this motor.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				void resetPosition();

				/**
				* Controls the brakes of this motor.
				*
				* @param brake	If set to TRUE, this will activate the brake. If set to FALSE, the brake is released.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				void setBrake( bool brake );

				/**
				* Sets the proportional, integral and  differential constant of the PID controller.
				* The range of values is from 0 to 255. These values are scaled by the microcontroller firmware
				* to match with the PID controller implementation.
				* If 255 is given, the microcontroller firmware uses its build in default value.
				* @param kp proportional constant. Typical value 200.
				* @param ki integral constant. Typical value 10.
				* @param kd differential constant. Typical value 0.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				void setPID( unsigned char kp, unsigned char ki, unsigned char kd );

				/**
				* Retrieves the actual speed of this motor.
				*
				* @return	Speed in rpm.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				float actualVelocity() const;

				/**
				* Retrieves the actual position of this motor.
				*
				* @return actual position
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				int actualPosition() const;

				/**
				* Retrieves the current of this motor.
				* @return motor current in A.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				float motorCurrent() const;

				/**
				* The current is measured by a 10 bit adc and is not converted into A.
				* @return The current delivered by to this motor.
				* Range from 0 to 1023 with I/O board version 1.x. I/O board version 2 uses a 12bit ADC. Range is then [0;4098].
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				unsigned short rawCurrentMeasurment() const;

			private:
				unsigned int _motorNumber;
			};
		}
	}
}
#endif

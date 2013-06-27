//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTSTATE_ENCODER_H_
#define _REC_ROBOTSTATE_ENCODER_H_

#include <string>
#include <map>
#include <boost/cstdint.hpp>

#include "rec/iocontrol/robotstate/State.h"

class QDSA_Encoder;

namespace rec
{
	namespace iocontrol
	{
		namespace robotstate
		{
			class Encoder
			{
			public:
				/**
				* Construct an Encoder for the give state
				*/
				Encoder( State* state );

				~Encoder();

				/**
				* Reset the state to default values
				*/
				void reset();

				/**
				* Set the velocity of the given motor.
				* This overrides any previous setVelocity calls for the given motor.
				* @param motor	The active motor, starting from 0.
				* @param rpm	The target speed in Rounds Per Minute
				* @throws	nothing.
				* @see		State::numMotors, Decoder::actualVelocity
				*/
				void setVelocity( unsigned int motor, float rpm );

				/**
				* RobotState::p2qUpdateCounter is not modified by this function
				@see setVelocity( unsigned int motor, float rpm )
				* @throws	nothing.
				*/
				void setVelocity_i( unsigned int motor, float rpm );

				/** 
				* Set the desired velocity of the robot.
				*
				* @param vX velocity in x direction in mm/s
				* @param vY velocity in y direction in mm/s
				* @param vOmega angular velocity in rad/s
				* @throws	nothing.
				*/
				void setVelocity( float vX, float vY, float vOmega );
				
				/**
				* RobotState::p2qUpdateCounter is not modified by this function
				@see setVelocity( float vX, float vY, float vOmega )
				* @throws	nothing.
				*/
				void setVelocity_i( float vX, float vY, float vOmega );

				/**
				* Stop all motors.
				*/
				void stopMotors();

				/**
				* Set the power output
				* @param setPoint Range -100 to 100
				*/
				void setPowerOutputSetPoint( float setPoint );

				/**
				* @param i Range from 0 to State::numDigitalOutputs
				*/
				void setDigitalOutput( unsigned int i, bool on );

				void setBrake( unsigned int motor, bool on );

				void resetPosition( unsigned int motor );

				void setRelay( unsigned int relay, bool on );

				/**
				* Shutdown Robotino. This will switch off power directly
				*/
				void setShutdown();

				/*
				* @param motor Range 0 to State::numMotors
				* @param kp Range 0.0 to 1.0
				* @param ki Range 0.0 to 1.0
				* @param kd Range 0.0 to 1.0
				*/
				void setPID( unsigned int motor, float kp, float ki, float kd );

				static void projectVelocity( float* m1, float* m2, float* m3, float vX, float vy, float omega, const DriveLayout& layout );

			private:
				State* _state;
				QDSA_Encoder* _enc;
			};
		}
	}
}

#endif //_REC_ROBOTSTATE_ENCODER_H_

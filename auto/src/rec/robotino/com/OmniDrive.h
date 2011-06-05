//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_OMNIRIVE_H_
#define _REC_ROBOTINO_COM_OMNIRIVE_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class OmniDriveImpl;

			/**
			* @brief	Calculates motor velocities for the omni drive.
			* 
			* Directional and rotational coordinates are provided and velocities
			* for the single motors are calculated.
			*
			* @see	Motor
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
			OmniDrive : public Actor
			{
			public:
				OmniDrive();
				
				virtual ~OmniDrive();

				/**
				* @param vx		Velocity in x-direction in mm/s
				* @param vy		Velocity in y-direction in mm/s
				* @param omega	Angular velocity in deg/s
				*/
				void setVelocity( float vx, float vy, float omega );

				/**
				* Project the velocity of the robot in cartesian coordinates to single motor speeds.
				*
				* @param m1		The resulting speed of motor 1 in rpm
				* @param m2		The resulting speed of motor 2 in rpm
				* @param m3		The resulting speed of motor 3 in rpm
				* @param vx		Velocity in x-direction in mm/s
				* @param vy		Velocity in y-direction in mm/s
				* @param omega	Angular velocity in deg/s
				* @throws		nothing.
				*/
				void project( float* m1, float* m2, float* m3, float vx, float vy, float omega ) const;

				/**
				* Project single motor speeds to velocity in cartesian coordinates.
				*
				* @param vx		The resulting speed in x-direction in mm/s
				* @param vy		The resulting speed in y-direction in mm/s
				* @param omega	The resulting angular velocity in deg/s
				* @param m1		Speed of motor 1 in rpm
				* @param m2		Speed of motor 2 in rpm
				* @param m3		Speed of motor 3 in rpm
				* @throws		nothing.
				*/
				void unproject( float* vx, float* vy, float* omega, float m1, float m2, float m3 ) const;

				/**
				* Sets the layout of the robots drive system in order to drive with mm/s. Default values are for Robotino.
				* @param rb		Distance from robot center to wheel center.
				* @param rw		Radius of the wheels.
				* @param fctrl	Frequency of control loop measuring the speed.
				* @param gear	Gear
				* @param mer	Motor encoder resolution.
				* @throws	nothing.
				* @see		getDriveLayout
				*/
				void setDriveLayout( double rb = 125.0f , double rw = 40.0, double fctrl = 900.0, double gear = 16, double mer = 2000 );

				/**
				* Retrieves the layout parameters of the robots drive system.
				* @param rb		Distance from robot center to wheel center.
				* @param rw		Radius of the wheels.
				* @param fctrl	Frequency of control loop measuring the speed.
				* @param gear	Gear
				* @param mer	Motor encoder resolution.
				* @throws	nothing.
				* @see		setDriveLayout
				*/
				void getDriveLayout( double* rb, double* rw, double* fctrl, double* gear, double* mer );

			private:
				void updateMotorSpeeds();

				OmniDriveImpl* _impl;
			};
		}
	}
}
#endif

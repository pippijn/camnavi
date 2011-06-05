//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_ODOMETRY_H_
#define _REC_ROBOTINO_COM_ODOMETRY_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	Represents Robotino's odoemtry module.
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
			Odometry : public Actor
			{
			public:
				Odometry();

				/**
				* @return Global x position of Robotino in mm.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				float x() const;

				/**
				* @return Global y position of Robotino in mm.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				float y() const;

				/**
				* @return Global orientation of Robotino in degree.
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				float phi() const;

				/**
				* Set Robotino's odoemtry to the given coordinates
				@param x Global x position in mm
				@param y Global y position in mm
				@param phi Global phi orientation in degrees
				* @throws	RobotinoException if the current communication object is invalid.
				*/
				void set( float x, float y, float phi );
			};
		}
	}
}
#endif //_REC_ROBOTINO_COM_ODOMETRY_H_



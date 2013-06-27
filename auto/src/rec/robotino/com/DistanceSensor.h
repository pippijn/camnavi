//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_DISTANCESENSOR_H_
#define _REC_ROBOTINO_COM_DISTANCESENSOR_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	Represents an IR distance sensor.
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
			DistanceSensor : public Actor
			{
			public:
				DistanceSensor();

				/**
				* @return Returns the number of distance sensors.
				*/
				static unsigned int numDistanceSensors();

				/**
				* Sets the number of this distance sensor.
				*
				* @param n	The input number. Range [0; Robotstate::numDistanceSensors]
				* @throws	RobotinoException if the given sensor number is invalid.
				*/
				void setSensorNumber( unsigned int n );

				/**
				* Returns the current voltage of this distance sensor. The voltage is correlated
				* to the measured distance, see the documentation for further details.
				* See http://www.acroname.com/robotics/info/articles/irlinear/irlinear.html for an explanation how to
				* convert the measured voltage into distance value.
				*
				* @return	The current voltage of this sensor.
				* @throws	RobotinoException if the underlying communication object is invalid
				* @see setSensorNumber, Actor::setComId
				*/
				float voltage() const;

				/**
				* Returns the heading of this distance sensor.
				* @return	The heading in degrees. [0; 360]
				* @throws	RobotinoException if the underlying communication object is invalid
				* @see setSensorNumber, Actor::setComId
				*/
				unsigned int heading() const;

			private:
				unsigned int _sensorNumber;
			};

		}
	}
}
#endif

//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_NORTHSTAR_H_
#define _REC_ROBOTINO_COM_NORTHSTAR_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	Represents the NorthStar tracking device.
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
			NorthStar : public Actor
			{
			public:
				NorthStar();

				/**
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				unsigned int sequenceNumber() const;

				/**
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				unsigned int roomId() const;

				/**
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				unsigned int numSpotsVisible() const;

				/**
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				float posX() const;

				/**
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				float posY() const;

				/**
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				float posTheta() const;

				/**
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				unsigned int magSpot0() const;

				/**
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				unsigned int magSpot1() const;

				/**
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				void setRoomId( char roomId );

				/**
				* @throws	RobotinoException if the underlying communication object is invalid
				*/
				void setCeilingCal( float ceilingCal );
			};
		}
	}
}
#endif

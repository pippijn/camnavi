//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_ACTOR_H_
#define _REC_ROBOTINO_COM_ACTOR_H_

#include "rec/robotino/com/ComId.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	The base class for all Robotino actors.
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
			Actor
			{
			public:
				Actor();
				virtual ~Actor();

				/** 
				* Sets the associated communication object.
				*
				* @param id The id of the associated communication object.
				* @throws	RobotinoException if given id is invalid.
				*/
				virtual void setComId( const ComId& id );

			protected:
				/**
				* The id of the Com object this actor is connected to.
				*/
				ComId _comID;
			};
		}
	}
}
#endif

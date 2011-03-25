//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_ENCODERINPUT_H_
#define _REC_ROBOTINO_COM_ENCODERINPUT_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	Represents the external motor encoder input.
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
			EncoderInput : public Actor
			{
			public:
				EncoderInput();

				/**
				Set the current position to zero.
				*/
				void resetPosition();

				/**
				@return Actual position in ticks since power on or resetPosition
				*/
				int position() const;

				/**
				@return The actual velocity in ticks/s
				*/
				int velocity() const;
			};
		}
	}
}
#endif //_REC_ROBOTINO_COM_ENCODERINPUT_H_



//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_JPGCAMERA_H_
#define _REC_ROBOTINO_COM_JPGCAMERA_H_

#include "rec/robotino/com/Actor.h"

namespace rec
{
	namespace robotino
	{     
		namespace com
		{
			/**
			* @brief	Represents a camera.
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
			JPGCamera : public Actor
			{
			public:
				JPGCamera();
				~JPGCamera();

				/**
				* Turns the streaming on and off.
				* Call this after connecting this object to a com object by calling setComId.
				*
				* @param streaming	TRUE to turn streaming on.
				* @throws	RobotinoException If an error occured.
				*/
				void setStreaming( bool streaming );

				/** 
				* Checks if streaming is enabled.
				*
				* @return	TRUE if streaming is enabled.
				* @throws	nothing.
				*/ 
				bool isStreaming() const;

				/**
				* Sets this camera to the given resolution (if possible)
				*
				* @param width Requested image width
				* @param height Requested image height
				* @throws	RobotinoException if given com object is invalid.
				*/
				void setResolution( unsigned int width, unsigned int height );

				/**
				* Returns the width and height set by setResolution
				* 
				* @param width Set to the currently requested image width
				* @param height Set to the currently requested image height
				* @throws	RobotinoException if given com object is invalid.
				*/
				void resolution( unsigned int* width, unsigned int* height ) const;

				/**
				* Called when an jpg or mjpeg image is received.
				* Note: If Com is not initialized with useQueuedCallback=true this function is called from outside the applications main thread.
				* This is extremely important particularly with regard to GUI applications.

				@param jpgData Contains the jpeg data. You must not delete this buffer. Instead you should make a copy of this buffer within this function call.
				@param jpgDataSize Is the size of the jpeg data buffer.
				@throws		nothing.
				@see Com::Com
				*/
				virtual void jpgReceivedEvent( const unsigned char* jpgData,
					                                unsigned int jpgDataSize );
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_JPGCAMERA_H_

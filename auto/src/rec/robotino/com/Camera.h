//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_CAMERA_H_
#define _REC_ROBOTINO_COM_CAMERA_H_

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
			Camera : public Actor
			{
			public:
				Camera();
				~Camera();

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
				* Called when an image is received.
				* Note: If Com is not initialized with useQueuedCallback=true this function is called from outside the applications main thread.
				* This is extremely important particularly with regard to GUI applications.

				@param imageData Contains the image data. You must not delete this buffer. Instead you should make a copy of this buffer within this function call.
				@param dataSize Is the size of the data buffer containing the uncompressed image.
				@param width The number of pixels per line
				@param height The number of lines
				@param numChannels The number of channels
				@param numBitsPerChannel The number of bits per channel
				@param step The number of bytes per line. This is normally width * numChannels.
				* @throws		nothing.
				*/
				virtual void imageReceivedEvent( const unsigned char* data,
					                             unsigned int dataSize,
												 unsigned int width,
												 unsigned int height,
												 unsigned int numChannels,
												 unsigned int bitsPerChannel,
												 unsigned int step );
			};
		}
	}
}

#endif

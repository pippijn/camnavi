//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/Camera.h"
#include "rec/robotino/com/RobotinoException.h"
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::Camera;
using rec::robotino::com::ComId;
using rec::robotino::com::RobotinoException;
using rec::robotino::com::ComImpl;

Camera::Camera()
{
}

Camera::~Camera()
{
	try
	{
		ComImpl *impl = ComImpl::instance( _comID );
		impl->deregisterStreamingCamera( this );
	}
	catch( const RobotinoException& )
	{
	}
}

void Camera::setResolution( unsigned int width, unsigned int height  )
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	if( impl->_setState.camera_imageWidth != width ||
			impl->_setState.camera_imageHeight != height )
	{
		impl->_setState.camera_imageWidth = width;
		impl->_setState.camera_imageHeight = height;
		impl->_cameraResolutionChanged = true;
	}
}

void Camera::resolution( unsigned int* width, unsigned int* height ) const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	*width = impl->_setState.camera_imageWidth;
	*height = impl->_setState.camera_imageHeight;
}

void Camera::setStreaming( bool streaming )
{
	ComImpl *impl = ComImpl::instance( _comID );

	if( streaming )
	{
		impl->registerStreamingCamera( this );
	}
	else
	{
		impl->deregisterStreamingCamera( this );
	}
}

bool Camera::isStreaming() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	Camera* p = const_cast<Camera*>( this );
	return impl->isStreaming( p );
}

void Camera::imageReceivedEvent( const unsigned char* data,
								 unsigned int dataSize,
								 unsigned int width,
								 unsigned int height,
								 unsigned int numChannels,
								 unsigned int bitsPerChannel,
								 unsigned int step )
{
}


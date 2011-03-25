//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/JPGCamera.h"
#include "rec/robotino/com/RobotinoException.h"
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::JPGCamera;
using rec::robotino::com::ComId;
using rec::robotino::com::RobotinoException;
using rec::robotino::com::ComImpl;

JPGCamera::JPGCamera()
{
}

JPGCamera::~JPGCamera()
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

void JPGCamera::setResolution( unsigned int width, unsigned int height  )
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

void JPGCamera::resolution( unsigned int* width, unsigned int* height ) const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	*width = impl->_setState.camera_imageWidth;
	*height = impl->_setState.camera_imageHeight;
}

void JPGCamera::setStreaming( bool streaming )
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

bool JPGCamera::isStreaming() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	JPGCamera* p = const_cast<JPGCamera*>( this );
	return impl->isStreaming( p );
}

void JPGCamera::jpgReceivedEvent( const unsigned char* jpgData,
					              unsigned int jpgDataSize )
{
}


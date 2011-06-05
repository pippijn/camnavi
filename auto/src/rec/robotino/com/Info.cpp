//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/Info.h"
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::Info;
using rec::robotino::com::ComId;
using rec::robotino::com::ComImpl;

Info::Info()
{
	try
	{
		ComImpl *impl = ComImpl::instance( _comID );
		impl->registerInfo( this );
	}
	catch( const RobotinoException& )
	{
		_comID = ComId::null;
	}
}

Info::~Info()
{
	try
	{
		ComImpl *impl = ComImpl::instance( _comID );
		impl->deregisterInfo( this );
	}
	catch( const RobotinoException& )
	{
	}
}

void Info::setComId( const ComId& id )
{
	if( id == _comID )
	{
		return;
	}

	try
	{
		ComImpl *impl = ComImpl::instance( _comID );
		impl->deregisterInfo( this );
	}
	catch( const RobotinoException& )
	{
	}

	// ComImpl::instance throws exception, if id is invalid.
	ComImpl *impl = ComImpl::instance( id );
	_comID = id;

	impl->registerInfo( this );
}

const char* Info::text() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_infoTextMutex );
	return impl->_infoText.c_str();
}


bool Info::isPassiveMode() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.isPassiveMode;
}

unsigned int Info::firmwareVersion() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.firmwareVersion;
}

void Info::infoReceivedEvent( const char* text )
{
}

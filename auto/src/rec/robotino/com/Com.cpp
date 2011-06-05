//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/Com.h"
#include "rec/robotino/com/ComImpl.hh"

using namespace rec::robotino::com;

Com::Com( bool useQueuedCallback )
: _impl( new ComImpl( this, useQueuedCallback ) )
{
}

Com::~Com()
{
	delete _impl;
}

ComId Com::id() const
{
	return _impl->comid;
}

void Com::connect( bool isBlocking )
{
	_impl->connect( isBlocking );
}

void Com::disconnect()
{
	_impl->disconnect();
}

void Com::setAddress( const char* address )
{
	_impl->setAddress( address );
}

const char* Com::address() const
{
	return _impl->address();
}

void Com::setImageServerPort( unsigned int port )
{
	_impl->setImageServerPort( port );
}

unsigned int Com::imageServerPort() const
{
	return _impl->imageServerPort();
}

bool Com::isConnected() const
{
	return _impl->isConnected();
}

Com::ConnectionState Com::connectionState() const
{
	return _impl->connectionState();
}

bool Com::isPassiveMode() const
{
	QMutexLocker lk( &_impl->_sensorStateMutex );
	return _impl->_sensorState.isPassiveMode;
}

void Com::processEvents()
{
	_impl->processEvents();
}

bool Com::hasPendingEvents() const
{
	return _impl->hasPendingEvents();
}

void Com::errorEvent( Error error, const char* errorString )
{
}

void Com::connectedEvent()
{
}

void Com::connectionClosedEvent()
{
}

void Com::connectionStateChangedEvent( ConnectionState newState, ConnectionState oldState )
{
}

void Com::updateEvent()
{
}

bool Com::waitForUpdate( unsigned int timeout )
{
	return _impl->waitForUpdate( timeout );
}

void Com::setMinimumUpdateCycleTime( unsigned int msecs )
{
	QMutexLocker lk( &_impl->_workerMutex );
	if( msecs > 100 )
	{
		msecs = 100;
	}

	_impl->_msecsPerUpdateCycle = msecs;
}

rec::iocontrol::remotestate::SensorState Com::sensorState()
{
	QMutexLocker lk( &_impl->_sensorStateMutex );
	return _impl->_sensorState;
}

void Com::setSetState( const rec::iocontrol::remotestate::SetState& setState )
{
	QMutexLocker lk( &_impl->_setStateMutex );

	if( setState.camera_imageWidth != _impl->_setState.camera_imageWidth
		|| setState.camera_imageHeight != _impl->_setState.camera_imageHeight )
	{
		_impl->_cameraResolutionChanged = true;
	}
	_impl->_setState = setState;
	_impl->_setState.imageServerPort = _impl->imageServerPort();
}

void Com::modeChangedEvent( bool isPassiveMode )
{
}

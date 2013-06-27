//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/Gripper.h"
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::Gripper;
using rec::robotino::com::ComId;
using rec::robotino::com::ComImpl;

Gripper::Gripper()
{
	try
	{
		ComImpl *impl = ComImpl::instance( _comID );
		impl->registerGripper( this );
	}
	catch( const RobotinoException& )
	{
		_comID = ComId::null;
	} 
}

Gripper::~Gripper()
{
	try
	{
		ComImpl *impl = ComImpl::instance( _comID );
		impl->deregisterGripper( this );
	}
	catch( const RobotinoException& )
	{
	}
}

void Gripper::setComId( const ComId& id )
{
	if( id == _comID )
	{
		return;
	}

	try
	{
		ComImpl *impl = ComImpl::instance( _comID );
		impl->deregisterGripper( this );
	}
	catch( const RobotinoException& )
	{
	}

	_comID = id;

	if( ComId::null != _comID )
	{
		ComImpl *impl = ComImpl::instance( _comID );
		impl->registerGripper( this );
	}
}

void Gripper::open()
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.gripper_close = false;
}

void Gripper::close()
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.gripper_close = true;
}

bool Gripper::isOpened() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.isGripperOpened;
}

bool Gripper::isClosed() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.isGripperClosed;
}

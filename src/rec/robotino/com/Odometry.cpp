//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/Odometry.h"
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::Odometry;
using rec::robotino::com::ComId;
using rec::robotino::com::ComImpl;

Odometry::Odometry()
{
}

float Odometry::x() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.odometryX;
}

float Odometry::y() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.odometryY;
}

float Odometry::phi() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.odometryPhi;
}

void Odometry::set( float x, float y, float phi )
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.setOdometry = true;
	impl->_setState.odometryX = x;
	impl->_setState.odometryY = y;
	impl->_setState.odometryPhi = phi;
}


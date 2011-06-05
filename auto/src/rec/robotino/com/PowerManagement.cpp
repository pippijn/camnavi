//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/PowerManagement.h"
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::PowerManagement;
using rec::robotino::com::ComImpl;

float PowerManagement::current() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.current;
}

float PowerManagement::voltage() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.voltage;
}

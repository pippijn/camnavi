//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/Bumper.h"
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::Bumper;
using rec::robotino::com::ComImpl;

bool Bumper::value() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.bumper;
}

//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/PowerOutput.h"
#include "rec/robotino/com/ComImpl.hh"
#include "rec/iocontrol/robotstate/State.h"

using rec::robotino::com::PowerOutput;
using rec::robotino::com::ComImpl;

PowerOutput::PowerOutput()
{
}

void PowerOutput::setValue( float controlPoint )
{
	ComImpl *impl = ComImpl::instance( _comID );

  short s;

	if( controlPoint >= 100.0f )
	{
		s = 255;
	}
	else if( controlPoint <= -100.0f )
	{
		s = -255;
	}
	else
	{
		s = static_cast<short>( 2.55f * controlPoint );
	}

	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.powerOutputControlPoint = s;
}

float PowerOutput::current() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.powerOutputCurrent;
}

unsigned short PowerOutput::rawCurrentMeasurment() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.powerOutputRawCurrent;
}
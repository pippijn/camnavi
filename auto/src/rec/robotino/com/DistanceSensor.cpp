//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/DistanceSensor.h"
#include "rec/robotino/com/ComImpl.hh"
#include "rec/robotino/com/RobotinoException.h"
#include "rec/iocontrol/robotstate/State.h"

using rec::robotino::com::DistanceSensor;
using rec::robotino::com::ComImpl;
using rec::robotino::com::RobotinoException;

DistanceSensor::DistanceSensor()
{
	_sensorNumber = 0;
}

unsigned int DistanceSensor::numDistanceSensors()
{
	return rec::iocontrol::robotstate::State::numDistanceSensors;
}

void DistanceSensor::setSensorNumber( unsigned int n )
{
	if( n >= numDistanceSensors() )
		throw RobotinoException( "Invalid sensor number" );

	_sensorNumber = n;
}

float DistanceSensor::voltage() const
{
	ComImpl *impl = ComImpl::instance( _comID );

	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.distanceSensor[_sensorNumber];
}

unsigned int DistanceSensor::heading() const
{
	return _sensorNumber * 40;
}

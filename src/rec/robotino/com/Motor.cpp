//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/Motor.h"
#include "rec/robotino/com/ComImpl.hh"
#include "rec/iocontrol/robotstate/State.h"

using rec::robotino::com::Motor;
using rec::robotino::com::ComImpl;

Motor::Motor()
: _motorNumber( 0 )
{
}

Motor::~Motor()
{
}

unsigned int Motor::numMotors()
{
	return 3;
}

void Motor::setMotorNumber( unsigned int n )
{
	if( n >= numMotors() )
		throw RobotinoException( "Invalid motor number" );

	_motorNumber = n;
}

void Motor::setSpeedSetPoint( float rpm )
{
	ComImpl *impl = ComImpl::instance( _comID );

	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.speedSetPoint[_motorNumber] = rpm;
}

void Motor::resetPosition( )
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	impl->resetPosition[_motorNumber] = true;
}

void Motor::setBrake( bool brake )
{
	ComImpl *impl = ComImpl::instance( _comID );
}

void Motor::setPID( unsigned char kp, unsigned char ki, unsigned char kd )
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.kp[_motorNumber] = kp;
	impl->_setState.ki[_motorNumber] = ki;
	impl->_setState.kd[_motorNumber] = kd;
}

float Motor::actualVelocity() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.actualVelocity[_motorNumber];
}

int Motor::actualPosition() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.actualPosition[_motorNumber];
}

float Motor::motorCurrent() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.motorCurrent[_motorNumber];
}

unsigned short Motor::rawCurrentMeasurment() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.rawMotorCurrent[_motorNumber];
}

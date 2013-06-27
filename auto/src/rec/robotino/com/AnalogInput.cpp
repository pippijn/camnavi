//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/AnalogInput.h"
#include "rec/robotino/com/ComImpl.hh"
#include "rec/robotino/com/RobotinoException.h"

using rec::robotino::com::AnalogInput;
using rec::robotino::com::ComImpl;
using rec::robotino::com::RobotinoException;

AnalogInput::AnalogInput()
: _inputNumber( 0 )
{
}

unsigned int AnalogInput::numAnalogInputs()
{
	return 8;
}

void AnalogInput::setInputNumber( unsigned int n )
{
	if( n >= numAnalogInputs() )
	{
		throw RobotinoException( "Invalid input number" );
	}

	_inputNumber = n;
}

float AnalogInput::value() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.aIn[_inputNumber]; 
}

//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/DigitalInput.h"
#include "rec/robotino/com/ComImpl.hh"
#include "rec/robotino/com/RobotinoException.h"

using rec::robotino::com::DigitalInput;
using rec::robotino::com::ComImpl;

DigitalInput::DigitalInput()
{
	_inputNumber = 0;
}

unsigned int DigitalInput::numDigitalInputs()
{
	return 8;
}

void DigitalInput::setInputNumber( unsigned int n )
{
	if( n >= numDigitalInputs() )
		throw RobotinoException( "Invalid input number" );

	_inputNumber = n;
}

bool DigitalInput::value() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.dIn[_inputNumber]; 
}

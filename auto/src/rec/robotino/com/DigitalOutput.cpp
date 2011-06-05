//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/DigitalOutput.h"
#include "rec/robotino/com/ComImpl.hh"
#include "rec/iocontrol/robotstate/State.h"

using rec::robotino::com::DigitalOutput;
using rec::robotino::com::ComImpl;

DigitalOutput::DigitalOutput()
{
	_outputNumber = 0;
}

unsigned int DigitalOutput::numDigitalOutputs()
{
	return rec::iocontrol::robotstate::State::numDigitalOutputs;
}

void DigitalOutput::setOutputNumber( unsigned int n )
{
	if( n >= numDigitalOutputs() )
		throw RobotinoException( "Invalid output number" );

	_outputNumber = n;
}

void DigitalOutput::setValue( bool on )
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.dOut[_outputNumber] = on;
}

//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/EncoderInput.h"
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::EncoderInput;
using rec::robotino::com::ComImpl;

EncoderInput::EncoderInput()
{
}

void EncoderInput::resetPosition()
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	impl->resetPosition[3] = true;
}

int EncoderInput::position() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.encoderInputPosition;
}

int EncoderInput::velocity() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.encoderInputVelocity;
}
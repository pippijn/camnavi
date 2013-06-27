//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/OmniDrive.h"
#include "rec/robotino/com/OmniDriveImpl.hh"
#include "rec/robotino/com/ComImpl.hh"
#include "rec/iocontrol/robotstate/Decoder.h"
#include "rec/iocontrol/robotstate/Encoder.h"

#include <math.h>

using rec::robotino::com::OmniDrive;
using rec::robotino::com::OmniDriveImpl;
using rec::robotino::com::ComImpl;

OmniDrive::OmniDrive()
: _impl( new OmniDriveImpl )
{
}

OmniDrive::~OmniDrive()
{
	delete _impl;
}

void OmniDrive::setDriveLayout( double rb , double rw, double fctrl, double gear, double mer )
{
	_impl->layout.rb = rb;
	_impl->layout.rw = rw;
	_impl->layout.fctrl = fctrl;
	_impl->layout.gear = gear;
	_impl->layout.mer = mer;
}

void OmniDrive::getDriveLayout( double* rb, double* rw, double* fctrl, double* gear, double* mer )
{
	*rb = _impl->layout.rb;
	*rw = _impl->layout.rw;
	*fctrl = _impl->layout.fctrl;
	*gear = _impl->layout.gear;
	*mer = _impl->layout.mer;
}

void OmniDrive::project( float* m1, float* m2, float* m3, float vx, float vy, float omega ) const
{
	rec::iocontrol::robotstate::Encoder::projectVelocity( m1, m2, m3, vx, vy, omega, _impl->layout );
}

void OmniDrive::unproject( float* vx, float* vy, float* omega, float m1, float m2, float m3 ) const
{
	rec::iocontrol::robotstate::Decoder::unprojectVelocity( vx, vy, omega, m1, m2, m3, _impl->layout );
}

void OmniDrive::setVelocity( float vx, float vy, float omega )
{
	ComImpl *impl = ComImpl::instance( _comID );

	float m1;
	float m2;
	float m3;

	project( &m1, &m2, &m3, vx, vy, omega );

	//conversion from rpm to inc is performed in SetState toQDSAProtocol
	//float s0 = m1 * static_cast<float>( _impl->layout.mer ) / static_cast<float>( _impl->layout.fctrl ) / static_cast<float>( 60.0 );
	//float s1 = m2 * static_cast<float>( _impl->layout.mer ) / static_cast<float>( _impl->layout.fctrl ) / static_cast<float>( 60.0 );
	//float s2 = m3 * static_cast<float>( _impl->layout.mer ) / static_cast<float>( _impl->layout.fctrl ) / static_cast<float>( 60.0 );

	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.speedSetPoint[0] = m1;
	impl->_setState.speedSetPoint[1] = m2;
	impl->_setState.speedSetPoint[2] = m3;
}

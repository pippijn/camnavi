//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/nstar/Com.h"
#include "rec/nstar/ComImpl.h"

#include <iostream>

using namespace rec::nstar;

Com::Com()
: _impl( new ComImpl( this ) )
{
}

Com::~Com()
{
	delete _impl;
}

bool Com::open( port_t port )
{
	return _impl->open( port );
}

bool Com::isOpen() const
{
	return _impl->isOpen();
}

void Com::close()
{
	_impl->close();
}

int Com::version() const
{
	return _impl->version();
}

const char* Com::portString() const
{
	return _impl->portString();
}

unsigned int Com::speed() const
{
	return _impl->speed();
}

bool Com::setReportFlags( bool report_pose,
						 unsigned int spot_report_mask,
						 unsigned int magnitude_report_mask,
						 unsigned int spot_avail_threshold )
{
	return _impl->setReportFlags( report_pose, spot_report_mask, magnitude_report_mask, spot_avail_threshold );
}

void Com::setRoom( unsigned int room )
{
	_impl->setRoom( room );
}

bool Com::setCeilingCal( float ceilingCal )
{
	return _impl->setCeilingCal( ceilingCal );
}

float Com::ceilingCal()
{
	return _impl->ceilingCal();
}

void Com::setRoomAndCeilingCal( unsigned int room, float ceilingCal )
{
	_impl->setRoomAndCeilingCal( room, ceilingCal );
}

bool Com::startContinuousReport()
{
	return _impl->startContinuousReport();
}

bool Com::stopContinuousReport()
{
	return _impl->stopContinuousReport();
}

bool Com::isContinousReportActive() const
{
	return _impl->isRunning();
}

bool Com::singleReport()
{
	return _impl->singleReport();
}

void Com::reportPoseEvent( const PoseReport& report )
{
}

void Com::reportSpotPositionEvent( const SpotPositionReport& report )
{
}

void Com::reportMagnitudeEvent( const MagnitudeReport& report )
{
}

void Com::reportError( const char* error )
{
}

void Com::continuousReportErrorEvent()
{
}


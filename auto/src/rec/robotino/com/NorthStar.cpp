//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/NorthStar.h"
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::NorthStar;
using rec::robotino::com::ComImpl;
using rec::robotino::com::ComId;

NorthStar::NorthStar()
{
}

unsigned int NorthStar::sequenceNumber() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.nstar_sequenceNumber;
}

unsigned int NorthStar::roomId() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.nstar_roomId;
}

unsigned int NorthStar::numSpotsVisible() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.nstar_numSpotsVisible;
}

float NorthStar::posX() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.nstar_posX;
}

float NorthStar::posY() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.nstar_posY;
}

float NorthStar::posTheta() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.nstar_posTheta;
}

unsigned int NorthStar::magSpot0() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.nstar_magSpot0;
}

unsigned int NorthStar::magSpot1() const
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_sensorStateMutex );
	return impl->_sensorState.nstar_magSpot1;
}

void NorthStar::setRoomId( char roomId )
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.nstar_roomId = roomId;
}

void NorthStar::setCeilingCal( float ceilingCal )
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.nstar_ceilingCal = ceilingCal;
}

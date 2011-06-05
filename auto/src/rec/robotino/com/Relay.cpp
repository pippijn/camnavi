//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/Relay.h"
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::Relay;
using rec::robotino::com::ComImpl;

Relay::Relay()
{
	_relayNumber = 0;
}

unsigned int Relay::numRelays()
{
	return 2;
}

void Relay::setRelayNumber( unsigned int n )
{
	if( n >= numRelays() )
		throw RobotinoException( "Invalid relay number" );

	_relayNumber = n;
}

void Relay::setValue( bool on )
{
	ComImpl *impl = ComImpl::instance( _comID );
	QMutexLocker lk( &impl->_setStateMutex );
	impl->_setState.relays[_relayNumber] = on;
}

//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/Actor.h"	
#include "rec/robotino/com/ComImpl.hh"

using rec::robotino::com::Actor;
using rec::robotino::com::ComId;
using rec::robotino::com::ComImpl;

/*
  Use ComID default constructor to get a valid ID
  so that Actors are associated to the first Com object by default
*/
Actor::Actor()
{
}

Actor::~Actor()
{
}

void Actor::setComId( const ComId& id )
{
	if( id == _comID )
	{
		return;
	}

	// ComImpl::instance throws exception, if id is invalid.
	ComImpl *impl = ComImpl::instance( id );
	_comID = id;
}

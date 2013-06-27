//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/ComId.h"

using namespace rec::robotino::com;

const ComId ComId::null = ComId( 0 );
unsigned int ComId::g_id = 1;

bool ComId::isNull() const
{
	return ( *this == null );
}

//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/utils.h"

#ifdef WIN32
#include <windows.h>
// _getch
#include <conio.h>
#else
// getchar
#include <stdio.h>
// usleep
#include <unistd.h>
#endif

void rec::core_lt::waitForKey()
{
#ifdef WIN32
	_getch();
#else
	::getchar();
#endif
}

void rec::core_lt::msleep( unsigned int ms )
{
#ifdef WIN32
	SleepEx( ms, false );
#else
	::usleep( ms * 1000 );
#endif
}

void rec::core_lt::msleep_HZ( unsigned int hz )
{
	if( 0 != hz )
	{
		rec::core_lt::msleep( 1000 / hz );
	}
}

//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/uconv.h"

#include <stdlib.h>
#include <boost/scoped_array.hpp>

std::wstring rec::core_lt::toWString( const std::string& str )
{
	std::wstring res;
	size_t s = ::mbstowcs( NULL, str.c_str(), str.size() );
	res.resize( s );
	::mbstowcs( &res[0], str.c_str(), str.size() );
	return res;
}

std::string rec::core_lt::toStdString( const std::wstring& ws )
{
	unsigned int maxSize = (ws.length() + 1) * 2;
	boost::scoped_array< char > buffer( new char[ maxSize ] );
	size_t s = ::wcstombs( buffer.get(), ws.c_str(), maxSize );
	if( s != (size_t) -1 )
	{
		// everything worked
		return std::string( buffer.get() );
	}
	else
	{
		return std::string();
	}
}

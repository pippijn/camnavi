//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/StringList.h"

rec::core_lt::StringList rec::core_lt::split( const std::string& source, const std::string& separator )
{
	rec::core_lt::StringList list;

	size_t lastPosOfSep = 0;
	size_t posOfSep = source.find( separator );
	for( ;; )
	{
		size_t length;
		if( std::string::npos != posOfSep )
		{
			length = posOfSep - lastPosOfSep;
		}
		else
		{
			length = std::string::npos;
		}

		std::string str = source.substr( lastPosOfSep, length );

		if( !str.empty() )
		{
			list.append( str );
		}

		if( std::string::npos == posOfSep )
		{
			break;
		}

		lastPosOfSep = posOfSep + separator.length();

		posOfSep = source.find( separator, lastPosOfSep );
	}

	return list;
}

rec::core_lt::StringList rec::core_lt::stringListFromString( const String& source )
{
	rec::core_lt::StringList list = rec::core_lt::split( source, ";;" );

	rec::core_lt::StringList::iterator iter = list.begin();
	while( list.end() != iter )
	{
		size_t posOfSemi = (*iter).find( "\\;" );
		while( std::string::npos != posOfSemi )
		{
			(*iter).replace( posOfSemi, 2, ";" );
			posOfSemi = (*iter).find( "\\;", posOfSemi+1 );
		}

		++iter;
	}

	return list;
}

rec::core_lt::String rec::core_lt::toString( const rec::core_lt::StringList& list )
{
	std::string output;

	rec::core_lt::StringList::const_iterator iter = list.constBegin();
	while( list.constEnd() != iter )
	{
		std::string str = *iter;

		size_t posOfSemi = str.find( ";" );
		while( std::string::npos != posOfSemi )
		{
			str.replace( posOfSemi, 1, "\\;" );
			posOfSemi = str.find( ";", posOfSemi+2 );
		}

		output += str;

		++iter;

		if( list.constEnd() != iter )
		{
			output += ";;";
		}
	}

	return output;
}



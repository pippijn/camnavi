//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/fileutils.h"

#include "rec/core_lt/Exception.h"
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/convenience.hpp>
#include <fstream>

std::string rec::core_lt::fileutils::readLineFromFile( const char* filename )
{
	std::ifstream is( filename, std::ios::in );
	if( ! is.is_open() )
	{
		throw rec::core_lt::Exception( std::string( "Could not open " ) + filename );
	}
	std::string str;
	is >> str;
	return str;
}

bool rec::core_lt::fileutils::isContainedIn( const std::string& baseDir, const std::string& relativeSubEntry )
{
	try
	{
		// TODO!!!
		return true;
		boost::filesystem::path bd( baseDir );
		boost::filesystem::path relPath( relativeSubEntry );
		boost::filesystem::path absPath = bd / relPath;
		boost::filesystem::path tmpPath = absPath.branch_path();
		if( boost::filesystem::equivalent( bd, tmpPath ) )
		{
			return true;
		}
	}
	catch( std::exception& )
	{
	}
	return false;
}

std::string rec::core_lt::fileutils::readFileToString( const std::string& filename )
{
	std::ifstream is( filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate );
	if( ! is.is_open() )
	{
		return "";
	}
	unsigned int size = is.tellg();
	std::string s;
	s.resize( size );
	is.seekg( 0, std::ios::beg );
	is.read( (char*) s.c_str(), size );
	is.close();
	return s;
}

/// only works with unix-like directory separators for crazy filenames
std::string rec::core_lt::fileutils::getExtension( const std::string& filename )
{
	std::string::size_type pos = filename.find_last_of( "/." );
	if( pos != std::string::npos )
	{
		if( filename.at( pos ) == '.' )
		{
			return filename.substr( pos + 1 );
		}
	}
	return std::string();
}

//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/core_lt/memory/FileIO.h"
#include <fstream>

using namespace rec::core_lt::memory;

ByteArray rec::core_lt::memory::read( const std::string& filename )
{
  std::ifstream is( filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate );
  if( !is.is_open() )
  {
    return ByteArray();
  }
  else
  {
    unsigned int size = is.tellg();
    is.seekg( 0, std::ios::beg );
		ByteArray ba( size );
    is.read( ba.data_s(), size );
    is.close();
    return ba;
  }
}
      
bool rec::core_lt::memory::write( const std::string& filename, const ByteArray& data )
{
	if( data.isEmpty() )
	{
		return false;
	}

  std::ofstream os( filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc );
  if( !os.is_open() )
  {
    return false;
  }
  os.write( data.constData_s(), data.size() );
  os.close();

  return true;
}

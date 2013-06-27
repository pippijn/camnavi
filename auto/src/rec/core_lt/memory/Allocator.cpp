//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/core_lt/memory/Allocator.h"
#include <cstring>

#include <stdlib.h>

using namespace rec::core_lt::memory;

unsigned char* Allocator::malloc( unsigned int size )
{
	return static_cast<unsigned char*>( ::malloc( size * sizeof( char ) ) );
}

unsigned char* Allocator::realloc( unsigned char* buffer, unsigned int newSize )
{
	return static_cast<unsigned char*>( ::realloc( buffer, newSize * sizeof( char ) ) );
}

void Allocator::free( unsigned char* p )
{
	::free( p );
}

void Allocator::copy( unsigned char* dst, const unsigned char* src, unsigned int size )
{
  memcpy( static_cast< void* >( dst ), static_cast< const void* >( src ), size );
}

bool Allocator::isAligned32( const unsigned char* const p )
{
  std::size_t i = reinterpret_cast< std::size_t >( p );
  return (i & 0x1F) == 0;
}

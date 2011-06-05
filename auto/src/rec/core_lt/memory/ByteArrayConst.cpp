//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/core_lt/memory/ByteArrayConst.h"
#include "rec/core_lt/memory/ByteArray.h"

#include <assert.h>
#include <fstream>
#include <cstring>

using namespace rec::core_lt::memory;

ByteArrayConst::ByteArrayConst()
: rec::core_lt::SharedBase< ByteArrayImpl >( new ByteArrayImpl )
{
}

ByteArrayConst::ByteArrayConst( const unsigned char* buffer, unsigned int length )
: rec::core_lt::SharedBase< ByteArrayImpl >( new ByteArrayImpl( buffer, length ) )
{
}

ByteArrayConst::ByteArrayConst( const char* buffer, unsigned int length )
: rec::core_lt::SharedBase< ByteArrayImpl >( new ByteArrayImpl( (const unsigned char*)buffer, length ) )
{
}

ByteArrayConst::ByteArrayConst( ByteArray& other )
: rec::core_lt::SharedBase< ByteArrayImpl >( new ByteArrayImpl )
{
	_impl = other._impl;
}

ByteArrayConst::ByteArrayConst( const ByteArray& other )
: rec::core_lt::SharedBase< ByteArrayImpl >( new ByteArrayImpl )
{
	_impl = other._impl;
}

ByteArrayConst::ByteArrayConst( ByteArrayImpl* impl )
: rec::core_lt::SharedBase< ByteArrayImpl >( impl )
{
}

ByteArrayConst ByteArrayConst::fromRawData( const unsigned char* data, unsigned int size )
{
	ByteArrayImpl* impl = new ByteArrayImpl;
	impl->data = const_cast<unsigned char*>( data );
	impl->size = size;
	impl->ownBuffer = false;
	return ByteArrayConst( impl );
}

ByteArrayConst ByteArrayConst::fromRawData( const char* data, unsigned int size )
{
	return fromRawData( (const unsigned char*)data, size );
}

ByteArrayConst ByteArrayConst::fromFile( const std::string& fileName )
{
	std::ifstream is( fileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate );
	if( is.is_open() )
	{
		unsigned int size = is.tellg();
		is.seekg( 0, std::ios::beg );

		ByteArray b( size );

		is.read( b.data_s(), b.size() );
		is.close();

		return b;
	}
	else
	{
		return ByteArrayConst();
	}
}

bool ByteArrayConst::isNull() const
{
	return NULL == _impl->data;
}

bool ByteArrayConst::isEmpty() const
{
	if( isNull() )
	{
		return true;
	}

	return ( 0 == _impl->size );
}

ByteArrayConst ByteArrayConst::deepCopy() const
{
	return ByteArrayConst( _impl->data, size() );
}

const unsigned char* ByteArrayConst::constData() const
{
	return _impl->data;
}

const char* ByteArrayConst::constData_s() const
{
	return reinterpret_cast< const char* >( _impl->data );
}

unsigned int ByteArrayConst::size() const
{
	return _impl->size;
}

unsigned int ByteArrayConst::bufferSize() const
{
	return _impl->dataSize;
}

bool ByteArrayConst::hasBufferOwnership() const
{
	return _impl->ownBuffer;
}

bool ByteArrayConst::operator!=( const ByteArrayConst& other ) const
{
	if( _impl->size != other._impl->size )
	{
		return true;
	}

	return ( 0 != memcmp( _impl->data, other._impl->data, _impl->size ) );
}

bool ByteArrayConst::operator==( const ByteArrayConst& other ) const
{
	if( _impl->size != other._impl->size )
	{
		return false;
	}

	return ( 0 == memcmp( _impl->data, other._impl->data, _impl->size ) );
}


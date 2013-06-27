//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include <rec/core_lt/memory/ByteArray.h>
#include <rec/core_lt/memory/DataStream.h>

#include <assert.h>
#include <cstring>

using namespace rec::core_lt::memory;

ByteArray::ByteArray()
{
}

ByteArray::ByteArray( unsigned int size )
: ByteArrayConst( new ByteArrayImpl( size ) )
{
}

ByteArray::ByteArray( const unsigned char* buffer, unsigned int length )
: ByteArrayConst( new ByteArrayImpl( buffer, length ) )
{
}

ByteArray::ByteArray( const char* buffer, unsigned int length )
: ByteArrayConst( new ByteArrayImpl( (const unsigned char*)buffer, length ) )
{
}

ByteArray::ByteArray( ByteArrayConst& other )
: ByteArrayConst( new ByteArrayImpl( other._impl->data, other._impl->size ) )
{
}

ByteArray::ByteArray( const ByteArrayConst& other )
: ByteArrayConst( new ByteArrayImpl( other._impl->data, other._impl->size ) )
{
}


ByteArray::ByteArray( ByteArrayImpl* impl )
: ByteArrayConst( impl )
{
}

ByteArray ByteArray::fromRawData( unsigned char* data, unsigned int size )
{
	ByteArrayImpl* impl = new ByteArrayImpl;
	impl->data = data;
	impl->size = size;
	impl->ownBuffer = false;
	return ByteArray( impl );
}

ByteArray ByteArray::fromRawData( char* data, unsigned int size )
{
	return fromRawData( (unsigned char*)data, size );
}

ByteArray ByteArray::fromFile( const std::string& fileName )
{
	return ByteArrayConst::fromFile( fileName );
}

ByteArray ByteArray::deepCopy() const
{
	return ByteArrayConst::deepCopy();
}

unsigned char* ByteArray::data()
{
	detach();
	return _impl->data;
}

char* ByteArray::data_s()
{
	detach();
	return reinterpret_cast< char* >( _impl->data );
}

void ByteArray::set( unsigned char value )
{
	detach();
	memset( static_cast< void* >( _impl->data ), value, _impl->size ); 
}

void ByteArray::resize( unsigned int size )
{
	if( size == _impl->size && _impl->ownBuffer )
	{
		return;
	}

	detach();

	unsigned int allignedNewDataSize = ((~(_impl->incrementSize-1)) & size) + _impl->incrementSize;

	if( allignedNewDataSize < _impl->dataSize )
	{
		_impl->data = Allocator::realloc( _impl->data, allignedNewDataSize );
		_impl->dataSize = allignedNewDataSize;
		_impl->size = size;
		return;
	}

	if( size < _impl->dataSize )
	{
		_impl->size = size;
		return;
	}

	_impl->data = Allocator::realloc( _impl->data, allignedNewDataSize );
	_impl->dataSize = allignedNewDataSize;
	_impl->size = size;
}

rec::core_lt::memory::ByteArray rec::core_lt::memory::replace( const rec::core_lt::memory::ByteArray& source, const std::string& what, const std::string& with )
{
	ByteArray target;
	DataStream os( &target );

	unsigned int i=0;
	while( i< source.size() )
	{
		const char* sourceData = source.constData_s() + i;

		if(
			i < source.size() - what.length() &&
			0 == strncmp( sourceData, what.c_str(), what.length() )
			)
		{
			os.enc( (const unsigned char*)with.c_str(), with.length() );
			i += what.length();
		}
		else
		{
			os.enc( (const unsigned char*)sourceData, 1 );
			++i;
		}
	}

	return target;
}

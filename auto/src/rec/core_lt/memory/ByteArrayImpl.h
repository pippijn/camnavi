//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_BYTEARRAYIMPL_H_
#define _REC_CORE_LT_MEMORY_BYTEARRAYIMPL_H_

#include <rec/core_lt/memory/Allocator.h>

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			class ByteArrayImpl
			{
			public:
				ByteArrayImpl()
					: size( 0 )
					, dataSize( 0 )
					, data( NULL )
					, ownBuffer( true )
					, incrementSize( 1024 )
				{
				}

				ByteArrayImpl( unsigned int size )
					: size( size )
					, dataSize( size )
					, data( rec::core_lt::memory::Allocator::malloc( dataSize ) )
					, ownBuffer( true )
					, incrementSize( 1024 )
				{
				}

				ByteArrayImpl( const unsigned char* buffer, unsigned int size )
					: size( size )
					, dataSize( size )
					, data( rec::core_lt::memory::Allocator::malloc( dataSize ) )
					, ownBuffer( true )
					, incrementSize( 1024 )
				{
					rec::core_lt::memory::Allocator::copy( data, buffer, size );
				}

				ByteArrayImpl( const ByteArrayImpl& src )
					: size( src.size )
					, dataSize( src.size )
					, data( rec::core_lt::memory::Allocator::malloc( dataSize ) )
					, ownBuffer( true )
					, incrementSize( 1024 )
				{
					rec::core_lt::memory::Allocator::copy( data, src.data, size );
				}

				~ByteArrayImpl()
				{
					if( NULL != data && ownBuffer )
					{
						rec::core_lt::memory::Allocator::free( data );
					}
				}

				//the size of the used data, which is <=dataSize
				unsigned int size;

				//the size of the allocated data
				unsigned int dataSize;

				unsigned char* data;

				bool ownBuffer;

				/**
				Number of bytes by which data is incremented on resize
				*/
				const unsigned int incrementSize;
			};
		}
	}
}

#endif //_REC_CORE_LT_MEMORY_BYTEARRAYIMPL_H_

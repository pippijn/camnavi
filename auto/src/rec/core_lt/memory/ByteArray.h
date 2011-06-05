//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_BYTEARRAY_H_
#define _REC_CORE_LT_MEMORY_BYTEARRAY_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/memory/ByteArrayConst.h"

#include <string>

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4251 )
#endif

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			class REC_CORE_LT_EXPORT ByteArray : public ByteArrayConst
			{
			public:
				ByteArray();
				ByteArray( unsigned int size );
				ByteArray( const unsigned char* data, unsigned int size );
				ByteArray( const char* data, unsigned int size );

				/**
				Copy data from other to this ByteArray.
				*/
				ByteArray( ByteArrayConst& other );

				/**
				Copy data from other to this ByteArray.
				*/
				ByteArray( const ByteArrayConst& other );

				/// Constructs a ByteArray that uses the first size characters in the array data.
				/// The bytes in data are not copied.
				/// The caller must be able to guarantee that data will not be deleted or modified as
				/// long as the ByteArray (or an unmodified copy of it) exists.
				/// Any attempts to modify the ByteArray or copies of it will cause it to create a deep
				/// copy of the data, ensuring that the raw data isn't modified.
				static ByteArray fromRawData( unsigned char* data, unsigned int size );
				
				static ByteArray fromRawData( char* data, unsigned int size );

				static ByteArray fromFile( const std::string& fileName );

				ByteArray deepCopy() const;

				/// Returns a pointer to the data stored in the byte array.
				/// The pointer can be used to access and modify the bytes that compose the array.
				/// Makes a copy of the data, if the data is shared with other copies of ByteArray
				unsigned char* data();

				/// This is an overloaded member function, provided for convenience.
				char* data_s();

				void set( unsigned char value );

				/// Lets the buffer grow to size. The buffer will not be reallocated if size is smaller than the current size.
				void resize( unsigned int size );

				/**
				* checks if buffer belongs to this object. If it doesn't, allocates new buffer and copies data
				*/
				void assureOwnBuffer();

				bool operator!=( const ByteArray& other ) const;
				
				bool operator==( const ByteArray& other ) const;

			protected:
				ByteArray( ByteArrayImpl* impl );
			};

			REC_CORE_LT_EXPORT rec::core_lt::memory::ByteArray replace( const rec::core_lt::memory::ByteArray& source, const std::string& what, const std::string& with );
		}
	}
}

#ifdef WIN32
#pragma warning( pop )
#endif

#endif

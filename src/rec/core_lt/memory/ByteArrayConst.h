//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_BYTEARRAYCONST_H_
#define _REC_CORE_LT_MEMORY_BYTEARRAYCONST_H_

#include "rec/core_lt/defines.h"
#include "rec/core_lt/SharedBase.h"

#include "rec/core_lt/memory/ByteArrayImpl.h"

#include <boost/shared_ptr.hpp>
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
			class ByteArray;

			class REC_CORE_LT_EXPORT ByteArrayConst : rec::core_lt::SharedBase< ByteArrayImpl >
			{
				friend class ByteArray;
			public:
				ByteArrayConst();
				ByteArrayConst( const unsigned char* data, unsigned int size );
				ByteArrayConst( const char* data, unsigned int size );
				ByteArrayConst( ByteArray& );
				ByteArrayConst( const ByteArray& );

				/// Constructs a ByteArray that uses the first size characters in the array data.
				/// The bytes in data are not copied.
				/// The caller must be able to guarantee that data will not be deleted or modified as
				/// long as the ByteArray (or an unmodified copy of it) exists.
				/// Any attempts to modify the ByteArray or copies of it will cause it to create a deep
				/// copy of the data, ensuring that the raw data isn't modified.
				static ByteArrayConst fromRawData( const unsigned char* data, unsigned int size );
				
				static ByteArrayConst fromRawData( const char* data, unsigned int size );

				static ByteArrayConst fromFile( const std::string& fileName );

				/// Returns true if this byte array is null; otherwise returns false.
				bool isNull() const;

				/// Returns true if the byte array has size 0; otherwise returns false.
				bool isEmpty() const;

				ByteArrayConst deepCopy() const;

				/// Returns a pointer to the data stored in the byte array.
				/// The pointer can be used to access the bytes that compose the array.
				/// The data is not copied.
				const unsigned char* constData() const;

				/// This is an overloaded member function, provided for convenience.
				const char* constData_s() const;

				unsigned int size() const;

				unsigned int bufferSize() const;

				/**
				* checks whether ByteArray is responsible for deallocation of buffer
				*/
				bool hasBufferOwnership() const;

				bool operator!=( const ByteArrayConst& other ) const;
				
				bool operator==( const ByteArrayConst& other ) const;

			protected:
				ByteArrayConst( ByteArrayImpl* );
			};
		}
	}
}

#ifdef WIN32
#pragma warning( pop )
#endif

#endif

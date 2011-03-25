//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_MARSHALL_DECODER_H_
#define _REC_CORE_LT_MEMORY_MARSHALL_DECODER_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/memory/ByteArrayConst.h"

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <string>

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			namespace marshall
			{
				class DecoderImpl;

				class Decoder : private boost::noncopyable
				{
				public:
					///@throws nothing
					Decoder( const ByteArrayConst& buffer, bool useBigEndian = true );

					~Decoder();

					///@throws nothing
					void reset();

					///@throws MarshallException
					void seek( unsigned int index );

					///@throws nothing
					unsigned int bytesLeft() const;

					///Reads at most len bytes from the stream into s and returns the number of bytes read. If an error occurs, an exception is thrown.
					///The buffer s must be preallocated. The data is not encoded.
					///@throws MarshallException
					void readRawData ( unsigned char* s, unsigned int len );

					///@throws MarshallException
					Decoder& operator>>( boost::uint8_t& value );
					
					///@throws MarshallException
					Decoder& operator>>( boost::uint16_t& value );
					
					///@throws MarshallException
					Decoder& operator>>( boost::uint32_t& value );
					
					///@throws MarshallException
					Decoder& operator>>( boost::uint64_t& value );
					
					///@throws MarshallException
					Decoder& operator>>( std::string& str );

					///@throws MarshallException
					Decoder& operator>>( boost::int8_t& value )
					{
						*this >> reinterpret_cast< boost::uint8_t& >( value );
						return *this;
					}

					///@throws MarshallException
					Decoder& operator>>( boost::int16_t& value )
					{
						*this >> reinterpret_cast< boost::uint16_t& >( value );
						return *this;
					}

					///@throws MarshallException
					Decoder& operator>>( boost::int32_t& value )
					{
						*this >> reinterpret_cast< boost::uint32_t& >( value );
						return *this;
					}

					///@throws MarshallException
					Decoder& operator>>( boost::int64_t& value )
					{
						*this >> reinterpret_cast< boost::uint64_t& >( value );
						return *this;
					}

				private:
					DecoderImpl* _impl;
				};

				class DecoderImpl
				{
				public:
					DecoderImpl( const ByteArrayConst& buffer )
						: buffer( buffer )
						, offset( 0 )
					{
					}

					virtual void operator>>( boost::uint8_t& value ) = 0;
					virtual void operator>>( boost::uint16_t& value ) = 0;
					virtual void operator>>( boost::uint32_t& value ) = 0;
					virtual void operator>>( boost::uint64_t& value ) = 0;

					const ByteArrayConst& buffer;
					unsigned int offset;
				};

			}
		}
	}
}

#endif

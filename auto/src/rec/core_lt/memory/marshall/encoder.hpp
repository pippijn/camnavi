//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_MARSHALL_ENCODER_H_
#define _REC_MARSHALL_ENCODER_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/memory/ByteArray.h"

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
				class EncoderImpl;

				class Encoder : private boost::noncopyable
				{
				public:
					///@throws nothing
					Encoder( ByteArray* buffer, bool useBigEndian = true );
					
					~Encoder();

					///@throws nothing
					void reset();
					
					///@throws nothing
					unsigned int offset() const;

					///@throws nothing
					void enc( const unsigned char* data, unsigned int length );

					///@throws nothing
					Encoder& operator<<( boost::uint8_t value );
					
					///@throws nothing
					Encoder& operator<<( boost::uint16_t value );
					
					///@throws nothing
					Encoder& operator<<( boost::uint32_t value );
					
					///@throws nothing
					Encoder& operator<<( boost::uint64_t value );
					
					///@throws nothing
					Encoder& operator<<( const std::string& str );

					///@throws nothing
					Encoder& operator<<( boost::int8_t value )
					{
						*this << static_cast< boost::uint8_t >( value );
						return *this;
					}
					
					///@throws nothing
					Encoder& operator<<( boost::int16_t value )
					{
						*this << static_cast< boost::uint16_t >( value );
						return *this;
					}
					
					///@throws nothing
					Encoder& operator<<( boost::int32_t value )
					{
						*this << static_cast< boost::uint32_t >( value );
						return *this;
					}
					
					///@throws nothing
					Encoder& operator<<( boost::int64_t value )
					{
						*this << static_cast< boost::uint64_t >( value );
						return *this;
					}

				private:
					EncoderImpl* _impl;
				};

				class EncoderImpl
				{
				public:
					EncoderImpl( ByteArray* buffer )
						: buffer( buffer )
						,offset( 0 )
					{
					}

					virtual void operator<<( boost::uint16_t value ) = 0;
					virtual void operator<<( boost::uint32_t value ) = 0;
					virtual void operator<<( boost::uint64_t value ) = 0;

					ByteArray* buffer;
					unsigned int offset;
				};

			}
		}
	}
}

#endif

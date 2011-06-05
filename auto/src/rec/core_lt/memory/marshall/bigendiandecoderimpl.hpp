//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_MARSHALL_BIGENDIANDECODERIMPL_H_
#define _REC_MARSHALL_BIGENDIANDECODERIMPL_H_

#include "rec/core_lt/memory/marshall/decoder.hpp"

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
				class BigEndianDecoderImpl : public DecoderImpl
				{
				public:
					BigEndianDecoderImpl( const ByteArrayConst& buffer )
						: DecoderImpl( buffer )
					{
					}

					///@throws MarshallException
					void operator>>( boost::uint8_t& value );
					
					///@throws MarshallException
					void operator>>( boost::uint16_t& value );
					
					///@throws MarshallException
					void operator>>( boost::uint32_t& value );
					
					///@throws MarshallException
					void operator>>( boost::uint64_t& value );

					static boost::uint8_t staticDecode8( const unsigned char* buffer );
					static boost::uint16_t staticDecode16( const unsigned char* buffer );
					static boost::uint32_t staticDecode32( const unsigned char* buffer );
					static boost::uint64_t staticDecode64( const unsigned char* buffer );
				};

			}
		}
	}
}

#endif

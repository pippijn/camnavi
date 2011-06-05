//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_MARSHALL_LITTLEENDIANENCODERIMPL_H_
#define _REC_MARSHALL_LITTLEENDIANENCODERIMPL_H_

#include "rec/core_lt/memory/marshall/encoder.hpp"
#include <boost/cstdint.hpp>

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			namespace marshall
			{
				class LittleEndianEncoderImpl : public EncoderImpl
				{
				public:
					///@throws nothing
					LittleEndianEncoderImpl( ByteArray* buffer )
						: EncoderImpl( buffer )
					{
					}

					///@throws nothing
					void operator<<( boost::uint16_t value );
					
					///@throws nothing
					void operator<<( boost::uint32_t value );
					
					///@throws nothing
					void operator<<( boost::uint64_t value );
				};
			}
		}
	}
}

#endif

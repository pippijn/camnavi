//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_MARSHALL_BIGENDIANENCODERIMPL_H_
#define _REC_MARSHALL_BIGENDIANENCODERIMPL_H_

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
				class BigEndianEncoderImpl : public EncoderImpl
				{
				public:
					BigEndianEncoderImpl( ByteArray* buffer )
						: EncoderImpl( buffer )
					{
					}

					void operator<<( boost::uint16_t value );
					void operator<<( boost::uint32_t value );
					void operator<<( boost::uint64_t value );
				};
			}
		}
	}
}

#endif

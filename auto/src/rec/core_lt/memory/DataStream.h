//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_DATASTREAM_H_
#define _REC_CORE_LT_MEMORY_DATASTREAM_H_

#include "rec/core_lt/defines.h"
#include "rec/core_lt/memory/ByteArrayConst.h"
#include "rec/core_lt/memory/ByteArray.h"

#include <boost/cstdint.hpp>

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			namespace marshall
			{
				class Encoder;
				class Decoder;
			}

			class REC_CORE_LT_EXPORT DataStream
			{
			public:
				typedef enum { StreamOk, EndOfStream, StreamError } Status;
				typedef enum { BigEndian, LittleEndian } ByteOrder;

				DataStream( ByteArray* data, ByteOrder byteOrder = BigEndian );
				DataStream( const ByteArrayConst& data, ByteOrder byteOrder = BigEndian );

				~DataStream();

				bool isOk() const { return ( StreamOk == _status ); }

				Status status() const { return _status; }
				void resetStatus();
				void setStatus( Status s );
				
				ByteOrder byteOrder() const { return _byteOrder; }

				int readRawData ( unsigned char* s, unsigned int len );

				DataStream& operator>>( boost::uint8_t& value );
				DataStream& operator>>( boost::int8_t& value );

				DataStream& operator>>( boost::uint16_t& value );
				DataStream& operator>>( boost::int16_t& value );

				DataStream& operator>>( boost::uint32_t& value );
				DataStream& operator>>( boost::int32_t& value );

				DataStream& operator>>( boost::uint64_t& value );
				DataStream& operator>>( boost::int64_t& value );

				DataStream& operator>>( std::string& str );

				void enc( const unsigned char* data, unsigned int length );

				DataStream& operator<<( boost::uint8_t value );
				DataStream& operator<<( boost::int8_t value );

				DataStream& operator<<( boost::uint16_t value );
				DataStream& operator<<( boost::int16_t value );

				DataStream& operator<<( boost::uint32_t value );
				DataStream& operator<<( boost::int32_t value );

				DataStream& operator<<( boost::uint64_t value );
				DataStream& operator<<( boost::int64_t value );

				DataStream& operator<<( const std::string& str );

			private:
				rec::core_lt::memory::marshall::Encoder* _enc;
				rec::core_lt::memory::marshall::Decoder* _dec;

				Status _status;
				ByteOrder _byteOrder;
			};
		}
	}
}

#endif //_REC_CORE_LT_MEMORY_DATASTREAM_H_

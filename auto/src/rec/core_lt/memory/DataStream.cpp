//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/core_lt/memory/DataStream.h"
#include "rec/core_lt/memory/marshall/decoder.hpp"
#include "rec/core_lt/memory/marshall/encoder.hpp"
#include "rec/core_lt/memory/marshall/marshallexception.hpp"

using rec::core_lt::memory::DataStream;
using rec::core_lt::memory::ByteArray;
using rec::core_lt::memory::ByteArrayConst;
using rec::core_lt::memory::marshall::Encoder;
using rec::core_lt::memory::marshall::Decoder;
using rec::core_lt::memory::marshall::MarshallException;

DataStream::DataStream( ByteArray* data, ByteOrder byteOrder )
: _enc( new Encoder( data, byteOrder == BigEndian ) )
, _dec( NULL )
, _status( StreamOk )
, _byteOrder( byteOrder )
{
}

DataStream::DataStream( const ByteArrayConst& data, ByteOrder byteOrder )
: _enc( NULL )
, _dec( new Decoder( data, byteOrder == BigEndian ) )
, _status( StreamOk )
, _byteOrder( byteOrder )
{
}

DataStream::~DataStream()
{
	delete _enc;
	delete _dec;
}

void DataStream::resetStatus()
{
	_status = StreamOk;
}

void DataStream::setStatus( Status s )
{
	_status = s;
}

int DataStream::readRawData ( unsigned char* s, unsigned int len )
{
	if( _dec )
	{
		try
		{
			_dec->readRawData( s, len );
			return len;
		}
		catch( const MarshallException& )
		{
			_status = EndOfStream;
			return -1;
		}
	}

	return -1;
}

DataStream& DataStream::operator>>( boost::uint8_t& value )
{
	if( _dec )
	{
		try
		{
			_dec->operator>>( value );
		}
		catch( const MarshallException& )
		{
			_status = EndOfStream;
		}
	}

	return *this;
}

DataStream& DataStream::operator>>( boost::int8_t& value )
{
	if( _dec )
	{
		try
		{
			_dec->operator>>( value );
		}
		catch( const MarshallException& )
		{
			_status = EndOfStream;
		}
	}

	return *this;
}

DataStream& DataStream::operator>>( boost::uint16_t& value )
{
	if( _dec )
	{
		try
		{
			_dec->operator>>( value );
		}
		catch( const MarshallException& )
		{
			_status = EndOfStream;
		}
	}

	return *this;
}

DataStream& DataStream::operator>>( boost::int16_t& value )
{
	if( _dec )
	{
		try
		{
			_dec->operator>>( value );
		}
		catch( const MarshallException& )
		{
			_status = EndOfStream;
		}
	}

	return *this;
}

DataStream& DataStream::operator>>( boost::uint32_t& value )
{
	if( _dec )
	{
		try
		{
			_dec->operator>>( value );
		}
		catch( const MarshallException& )
		{
			_status = EndOfStream;
		}
	}

	return *this;
}

DataStream& DataStream::operator>>( boost::int32_t& value )
{
	if( _dec )
	{
		try
		{
			_dec->operator>>( value );
		}
		catch( const MarshallException& )
		{
			_status = EndOfStream;
		}
	}

	return *this;
}

DataStream& DataStream::operator>>( boost::uint64_t& value )
{
	if( _dec )
	{
		try
		{
			_dec->operator>>( value );
		}
		catch( const MarshallException& )
		{
			_status = EndOfStream;
		}
	}

	return *this;
}

DataStream& DataStream::operator>>( boost::int64_t& value )
{
	if( _dec )
	{
		try
		{
			_dec->operator>>( value );
		}
		catch( const MarshallException& )
		{
			_status = EndOfStream;
		}
	}

	return *this;
}

DataStream& DataStream::operator>>( std::string& str )
{
	if( _dec )
	{
		try
		{
			_dec->operator>>( str );
		}
		catch( const MarshallException& )
		{
			_status = EndOfStream;
		}
	}

	return *this;
}

void DataStream::enc( const unsigned char* data, unsigned int length )
{
	if( _enc )
	{
		_enc->enc( data, length );
	}
}

DataStream& DataStream::operator<<( boost::uint8_t value )
{
	if( _enc )
	{
		_enc->operator<<( value );
	}

	return *this;
}

DataStream& DataStream::operator<<( boost::int8_t value )
{
	if( _enc )
	{
		_enc->operator<<( value );
	}

	return *this;
}

DataStream& DataStream::operator<<( boost::uint16_t value )
{
	if( _enc )
	{
		_enc->operator<<( value );
	}

	return *this;
}

DataStream& DataStream::operator<<( boost::int16_t value )
{
	if( _enc )
	{
		_enc->operator<<( value );
	}

	return *this;
}

DataStream& DataStream::operator<<( boost::uint32_t value )
{
	if( _enc )
	{
		_enc->operator<<( value );
	}

	return *this;
}

DataStream& DataStream::operator<<( boost::int32_t value )
{
	if( _enc )
	{
		_enc->operator<<( value );
	}

	return *this;
}

DataStream& DataStream::operator<<( boost::uint64_t value )
{
	if( _enc )
	{
		_enc->operator<<( value );
	}

	return *this;
}

DataStream& DataStream::operator<<( boost::int64_t value )
{
	if( _enc )
	{
		_enc->operator<<( value );
	}

	return *this;
}

DataStream& DataStream::operator<<( const std::string& str )
{
	if( _enc )
	{
		_enc->operator<<( str );
	}

	return *this;
}

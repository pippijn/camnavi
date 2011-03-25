//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/memory/marshall/decoder.hpp"
#include "rec/core_lt/memory/marshall/marshallexception.hpp"
#include "rec/core_lt/memory/marshall/bigendiandecoderimpl.hpp"
#include "rec/core_lt/memory/marshall/littleendiandecoderimpl.hpp"

#include <cstring>

using rec::core_lt::memory::marshall::Decoder;
using rec::core_lt::memory::ByteArrayConst;

Decoder::Decoder( const ByteArrayConst& buffer, bool useBigEndian )
: _impl( NULL )
{
  if( useBigEndian )
  {
    _impl = new rec::core_lt::memory::marshall::BigEndianDecoderImpl( buffer );
  }
  else
  {
    _impl = new rec::core_lt::memory::marshall::LittleEndianDecoderImpl( buffer );
  }
}

Decoder::~Decoder()
{
  delete _impl;
}

void Decoder::reset()
{
  _impl->offset = 0;
}

void Decoder::seek( unsigned int index )
{
  if( index >= _impl->buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
  _impl->offset = index;
}

unsigned int Decoder::bytesLeft() const
{
  return _impl->buffer.size() - _impl->offset;
}

void Decoder::readRawData( unsigned char* buffer, unsigned int length )
{
  if( _impl->offset+length > _impl->buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
  memcpy( static_cast< void* >( buffer ), static_cast< const void* >( _impl->buffer.constData() + _impl->offset ), length );
  _impl->offset += length;
}


Decoder& Decoder::operator>>( boost::uint8_t& value )
{
  _impl->operator>>( value );
  return *this;
}

Decoder& Decoder::operator>>( boost::uint16_t& value )
{
  _impl->operator>>( value );
  return *this;
}

Decoder& Decoder::operator>>( boost::uint32_t& value )
{
  _impl->operator>>( value );
  return *this;
}

Decoder& Decoder::operator>>( boost::uint64_t& value )
{
  _impl->operator>>( value );
  return *this;
}

Decoder& Decoder::operator>>( std::string& str )
{
	str = std::string( reinterpret_cast< const char* >( _impl->buffer.constData_s() +_impl->offset ) );
  if( _impl->offset + str.size() > _impl->buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
  _impl->offset += ( str.size() + 1 );
  return *this;
}


//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/memory/marshall/encoder.hpp"
#include "rec/core_lt/memory/marshall/bigendianencoderimpl.hpp"
#include "rec/core_lt/memory/marshall/littleendianencoderimpl.hpp"
#include "rec/core_lt/memory/marshall/marshallexception.hpp"

#include <cstring>

using rec::core_lt::memory::ByteArray;
using rec::core_lt::memory::marshall::Encoder;
using rec::core_lt::memory::marshall::EncoderImpl;

Encoder::Encoder( ByteArray* buffer, bool useBigEndian )
: _impl( NULL )
{
  if( useBigEndian )
  {
    _impl = new rec::core_lt::memory::marshall::BigEndianEncoderImpl( buffer );
  }
  else
  {
    _impl = new rec::core_lt::memory::marshall::LittleEndianEncoderImpl( buffer );
  }
}

Encoder::~Encoder()
{
  delete _impl;
}

void Encoder::reset()
{
  _impl->offset = 0;
}

unsigned int Encoder::offset() const
{
  return _impl->offset;
}

void Encoder::enc( const unsigned char* data, unsigned int length )
{
	_impl->buffer->resize( _impl->offset+length );
  memcpy( static_cast< void* >( _impl->buffer->data()+_impl->offset ), static_cast< const void* >( data ), length );
  _impl->offset += length;
}

Encoder& Encoder::operator<<( boost::uint8_t value )
{
	_impl->buffer->resize( _impl->offset+1 );
  _impl->buffer->data()[ _impl->offset++ ] = value;
  return *this;
}

Encoder& Encoder::operator<<( boost::uint16_t value )
{
  _impl->operator<<( value );
  return *this;
}

Encoder& Encoder::operator<<( boost::uint32_t value )
{
  _impl->operator<<( value );
  return *this;
}

Encoder& Encoder::operator<<( boost::uint64_t value )
{
  _impl->operator<<( value );
  return *this;
}

Encoder& Encoder::operator<<( const std::string& str )
{
	_impl->buffer->resize( _impl->offset+str.size()+1 );
  memcpy( static_cast< void* >( _impl->buffer->data()+_impl->offset ), static_cast< const void* >( str.c_str() ), str.size()+1 );
  _impl->offset += str.size()+1;
  
  return *this;
}

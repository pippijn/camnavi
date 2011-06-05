//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/memory/marshall/decoder.hpp"
#include "rec/core_lt/memory/marshall/bigendiandecoderimpl.hpp"
#include "rec/core_lt/memory/marshall/marshallexception.hpp"

using rec::core_lt::memory::marshall::BigEndianDecoderImpl;

void BigEndianDecoderImpl::operator>>( boost::uint8_t& value )
{
  if( offset+1 > buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
	value = staticDecode8( buffer.constData() + offset );
  offset += 1;
}

void BigEndianDecoderImpl::operator>>( boost::uint16_t& value )
{
  if( offset+2 > buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
  value = staticDecode16( buffer.constData() + offset );
  offset += 2;
}

void BigEndianDecoderImpl::operator>>( boost::uint32_t& value )
{
  if( offset+4 > buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
  value = staticDecode32( buffer.constData() + offset );
  offset += 4;
}

void BigEndianDecoderImpl::operator>>( boost::uint64_t& value )
{
  if( offset+8 > buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
  value = staticDecode64( buffer.constData() + offset );  
  offset += 8;
}

boost::uint8_t BigEndianDecoderImpl::staticDecode8( const unsigned char* buffer )
{
  boost::uint8_t value = buffer[ 0 ];
  return value;
}

boost::uint16_t BigEndianDecoderImpl::staticDecode16( const unsigned char* buffer )
{
  return (((boost::uint16_t) buffer[0] << 8) | buffer[1]);
}

boost::uint32_t BigEndianDecoderImpl::staticDecode32( const unsigned char* buffer )
{
  boost::uint32_t value = (((boost::uint32_t) buffer[2] << 8) | buffer[3]);
  value = value | ((boost::uint32_t) buffer[1] << 16);
  value = value | ((boost::uint32_t) buffer[0] << 24);
  return value;
}

boost::uint64_t BigEndianDecoderImpl::staticDecode64( const unsigned char* buffer )
{
  boost::uint64_t value = staticDecode32( buffer+4 );
  value = value | ( static_cast< boost::uint64_t >( staticDecode32( buffer ) ) << 32 );  
  return value;
}

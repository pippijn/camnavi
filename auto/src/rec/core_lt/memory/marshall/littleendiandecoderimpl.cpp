//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/memory/marshall/decoder.hpp"
#include "rec/core_lt/memory/marshall/littleendiandecoderimpl.hpp"
#include "rec/core_lt/memory/marshall/marshallexception.hpp"

using rec::core_lt::memory::marshall::LittleEndianDecoderImpl;

void LittleEndianDecoderImpl::operator>>( boost::uint8_t& value )
{
  if( offset+1 > buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
  value = staticDecode8( buffer.constData() + offset );
  offset += 1;
}

void LittleEndianDecoderImpl::operator>>( boost::uint16_t& value )
{
  if( offset+2 > buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
  value = staticDecode16( buffer.constData() + offset );
  offset += 2;
}

void LittleEndianDecoderImpl::operator>>( boost::uint32_t& value )
{
  if( offset+4 > buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
  value = staticDecode32( buffer.constData() + offset );
  offset += 4;
}

void LittleEndianDecoderImpl::operator>>( boost::uint64_t& value )
{
  if( offset+8 > buffer.size() )
  {
    throw rec::core_lt::memory::marshall::MarshallException( "Buffer overflow" );
  }
  value = staticDecode64( buffer.constData() + offset );  
  offset += 8;
}

boost::uint8_t LittleEndianDecoderImpl::staticDecode8( const unsigned char* buffer )
{
  boost::uint8_t value = buffer[ 0 ];
  return value;
}

boost::uint16_t LittleEndianDecoderImpl::staticDecode16( const unsigned char* buffer )
{
  return (((boost::uint16_t) buffer[1] << 8) | buffer[0]);
}

boost::uint32_t LittleEndianDecoderImpl::staticDecode32( const unsigned char* buffer )
{
  boost::uint32_t value = (((boost::uint32_t) buffer[1] << 8) | buffer[0]);
  value = value | ((boost::uint32_t) buffer[2] << 16);
  value = value | ((boost::uint32_t) buffer[3] << 24);
  return value;
}

boost::uint64_t LittleEndianDecoderImpl::staticDecode64( const unsigned char* buffer )
{
  boost::uint64_t value = staticDecode32( buffer );
  value = value | ( static_cast< boost::uint64_t >( staticDecode32( buffer+4 ) ) << 32 );  
  return value;
}

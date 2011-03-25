//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/memory/marshall/bigendianencoderimpl.hpp"

using rec::core_lt::memory::marshall::Encoder;
using rec::core_lt::memory::marshall::BigEndianEncoderImpl;


void BigEndianEncoderImpl::operator<<( boost::uint16_t value )
{
	buffer->resize( offset+2 );
  buffer->data()[ offset++ ] = (value >> 8) & 0xFF;
  buffer->data()[ offset++ ] = value & 0xFF;
}

void BigEndianEncoderImpl::operator<<( boost::uint32_t value )
{
	buffer->resize( offset+4 );
  buffer->data()[ offset++ ] = static_cast< unsigned char>((value >> 24) & 0xFF );
  buffer->data()[ offset++ ] = static_cast< unsigned char>((value >> 16) & 0xFF );
  buffer->data()[ offset++ ] = static_cast< unsigned char>( (value >> 8) & 0xFF );
  buffer->data()[ offset++ ] = static_cast< unsigned char>(value & 0xFF);
}

void BigEndianEncoderImpl::operator<<( boost::uint64_t value )
{
  *this << static_cast< boost::uint32_t >( value >> 32 );
  *this << static_cast< boost::uint32_t >( value & 0xFFFFFFFF );
}

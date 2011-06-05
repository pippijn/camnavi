//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/iocontrol/sercom/sercomencoder.h"
#include <memory.h>
#include <assert.h>

SercomEncoder::SercomEncoder( unsigned int p2qSize, unsigned int q2pSize, bool bitMode )
: _p2q_size( p2qSize )
, _q2p_size( q2pSize )
, _isBufferOwner( true )
, _bitMode( bitMode )
{
  _p2q_buffer = new unsigned char[ _p2q_size ];
  _q2p_buffer = new unsigned char[ _q2p_size ];
}

SercomEncoder::SercomEncoder( unsigned char* p2q_buffer, unsigned int p2qSize, unsigned char* q2p_buffer, unsigned int q2pSize, bool bitMode )
: _p2q_size( p2qSize )
, _q2p_size( q2pSize )
, _isBufferOwner( false )
, _bitMode( bitMode )
, _p2q_buffer( p2q_buffer )
, _q2p_buffer( q2p_buffer )
{
}

SercomEncoder::~SercomEncoder()
{
	if( _isBufferOwner )
	{
		delete [] _p2q_buffer;
		delete [] _q2p_buffer;
	}
}

void SercomEncoder::setBit( unsigned char* byte, unsigned int bit, unsigned int value )
{
  if( value > 0 )
  {
    *byte |= _BV( bit );
  }
  else
  {
    *byte &= ~_BV( bit );
  }
}

void SercomEncoder::clone( const SercomEncoder* enc, SercomEncoder::CloneType ct )
{
  assert( _p2q_size == enc->_p2q_size );
  assert( _q2p_size == enc->_q2p_size );
  lock lk1( _p2q_mutex );
  lock lk2( _q2p_mutex );
  if( ct & SercomEncoder::CLONE_P2Q )
  {
    memcpy( static_cast<void*>(_p2q_buffer), (const void*) enc->_p2q_buffer, _p2q_size );
  }
  if( ct & SercomEncoder::CLONE_Q2P )
  {
    memcpy( static_cast<void*>(_q2p_buffer), (const void*) enc->_q2p_buffer, _q2p_size );
  }
}

unsigned int SercomEncoder::p2qSize() const
{
  return _p2q_size;
}

unsigned int SercomEncoder::q2pSize() const
{
  return _q2p_size;
}

void SercomEncoder::set_q2p( const unsigned char* data )
{
  lock lk( _q2p_mutex );
  memcpy( (void*) _q2p_buffer, (const void*) data, _q2p_size ); 
}

void SercomEncoder::set_p2q( const unsigned char* data )
{
  lock lk( _p2q_mutex );
  memcpy( (void*) _p2q_buffer, (const void*) data, _p2q_size ); 
}

void SercomEncoder::get_p2q( unsigned char* buffer )
{
  lock lk( _p2q_mutex );
  memcpy( (void*) buffer, (const void*) _p2q_buffer, _p2q_size ); 
}

void SercomEncoder::get_q2p( unsigned char* buffer )
{
  lock lk( _q2p_mutex );
  memcpy( (void*) buffer, (const void*) _q2p_buffer, _q2p_size ); 
}

const unsigned char* SercomEncoder::p2q() const
{
  return _p2q_buffer;
}

const unsigned char* SercomEncoder::q2p() const
{
  return _q2p_buffer;
}

void SercomEncoder::preSend()
{
}

void SercomEncoder::postSendOk()
{
}

void SercomEncoder::dataReceived()
{
}

bool SercomEncoder::operator!=( const SercomEncoder& c ) const
{
  return !( *this == c );
}

bool SercomEncoder::operator==( const SercomEncoder& c ) const
{
  if( _p2q_size != c._p2q_size )
  {
    return false;
  }
  if( strncmp( (const char*)_p2q_buffer, (const char*)c._p2q_buffer, _p2q_size ) != 0 )
  {
    return false;
  }
  if( _q2p_size != c._q2p_size )
  {
    return false;
  }
  if( strncmp( (const char*)_q2p_buffer, (const char*)c._q2p_buffer, _q2p_size ) != 0 )
  {
    return false;
  }

  return true;
}

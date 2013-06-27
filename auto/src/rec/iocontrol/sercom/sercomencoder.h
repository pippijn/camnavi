//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _SERCOMENCODER_H_
#define _SERCOMENCODER_H_

#include <boost/thread/mutex.hpp>
#include <string>
#include <map>

#define _BV(bit) (1 << (bit))
#define bit_is_set(sfr, bit) ( (sfr & _BV(bit)) > 0 ? true : false )

class SercomEncoder
{
public:
  typedef enum { CLONE_P2Q = 1, CLONE_Q2P = 2, CLONE_ALL = 3 } CloneType;
  typedef boost::mutex::scoped_lock lock;

  SercomEncoder( unsigned int p2qSize, unsigned int q2pSize, bool bitMode = false );
  
	SercomEncoder( unsigned char* p2q_buffer, unsigned int p2qSize, unsigned char* q2p_buffer, unsigned int q2pSize, bool bitMode = false );
  
	virtual ~SercomEncoder();

  virtual void clone( const SercomEncoder* enc, CloneType ct = CLONE_ALL );

  virtual void reset() = 0;
  virtual unsigned char ackChar() const = 0;
  virtual unsigned char restartChar() const = 0;
  virtual unsigned int bytesPerPacketP2Q() const = 0;
  virtual unsigned int bytesPerPacketQ2P() const = 0;
  virtual void preSend();
  virtual void postSendOk();
  virtual void dataReceived();

  // checks start and stop sequence
  virtual bool checkQ2P( const unsigned char* buffer, unsigned int relevantByte ) = 0;

  unsigned int p2qSize() const;
  unsigned int q2pSize() const;

  void set_q2p( const unsigned char* data );
  void set_p2q( const unsigned char* data );
  void get_p2q( unsigned char* buffer );
  void get_q2p( unsigned char* buffer );
  const unsigned char* p2q() const;
  const unsigned char* q2p() const;

  bool operator==( const SercomEncoder& ) const;
  bool operator!=( const SercomEncoder& ) const;

	const bool isBufferOwner() const { return _isBufferOwner; }

protected:
  void setBit( unsigned char* byte, unsigned int bit, unsigned int value );
  
  const unsigned int _p2q_size;
  const unsigned int _q2p_size;

  unsigned char* _p2q_buffer;
  unsigned char* _q2p_buffer;

  mutable boost::mutex _p2q_mutex;
  mutable boost::mutex _q2p_mutex;

  // true: veroderung der werte
  bool _bitMode;

	const bool _isBufferOwner;
};

#endif


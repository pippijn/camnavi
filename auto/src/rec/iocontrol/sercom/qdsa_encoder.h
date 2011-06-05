//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _QDSA_ENCODER_H_
#define _QDSA_ENCODER_H_

#include <string>
#include <map>
#include <list>

#include <boost/cstdint.hpp>

#include "rec/iocontrol/sercom/sercomencoder.h"

class PortIdentifier
{
public:
  PortIdentifier() : _byte( 0 ), _bit( 0 ), _initial( 0 ) {}
  PortIdentifier( unsigned int byte, unsigned int bit, unsigned int init = 0 ) : _byte( byte ), _bit( bit ), _initial( init ) {}
  PortIdentifier( const PortIdentifier& pi ) : _byte( pi._byte ), _bit( pi._bit ), _initial( pi._initial ) {}
  PortIdentifier& operator=( const PortIdentifier& pi )
  {
    _byte = pi._byte;
    _bit = pi._bit;
    _initial = pi._initial;
    return *this;
  }

  unsigned int byte() const { return _byte; }
  unsigned int bit() const { return _bit; }
  unsigned int initial() const { return _initial; }
  
private:
  unsigned int _byte;
  unsigned int _bit;
  unsigned int _initial;
};

class ADIdentifier
{
public:
  ADIdentifier() : _hibyte( 0 ), _lobyte( 0 ), _channel( 0 ) {}
  ADIdentifier( unsigned int hibyte, unsigned int lobyte, unsigned int channel ) : _hibyte( hibyte ), _lobyte( lobyte ), _channel( channel ) {}
  ADIdentifier( const ADIdentifier& pi ) : _hibyte( pi._hibyte ), _lobyte( pi._lobyte ), _channel( pi._channel ) {}
  ADIdentifier& operator=( const ADIdentifier& pi )
  {
    _hibyte = pi._hibyte;
    _lobyte = pi._lobyte;
    _channel = pi._channel;
    return *this;
  }

  unsigned int hibyte() const { return _hibyte; }
  unsigned int lobyte() const { return _lobyte; }
  unsigned int channel() const { return _channel; }
  
private:
  unsigned int _hibyte;
  unsigned int _lobyte;
  unsigned int _channel;
};


class QDSA_Encoder : public SercomEncoder
{
public:
  enum QDSA_Encoder_Error{NoError,SPI0Error,SPI1Error};

  static const unsigned int numMotors = 4;

	/**
	* This motor is a power output without PID controller
	*/
  static const unsigned int powerOutput = 3;

  static const unsigned int numDistances = 9;
  static const unsigned int numBumpers = 1;
  static const unsigned int numDO = 8;
  static const unsigned int numDI = 8;
  static const unsigned int numADC = 8;
  static const unsigned int numRelays = 2;

  QDSA_Encoder( bool bitMode = false );
  
	QDSA_Encoder( unsigned char* p2q_buffer, unsigned int p2qSize, unsigned char* q2p_buffer, unsigned int q2pSize, bool bitMode = false );

  QDSA_Encoder* deepCopy() const;

  void reset();

  virtual unsigned char ackChar() const;
  virtual unsigned char restartChar() const;
  virtual unsigned int bytesPerPacketQ2P() const;
  virtual unsigned int bytesPerPacketP2Q() const;
  virtual bool checkQ2P( const unsigned char* buffer, unsigned int relevantByte );
  
  void set_DO( const std::string& name, boost::uint8_t value );
  bool get_DO( const std::string& name ) const;

  void set_DV( boost::uint8_t motor, boost::int16_t value );
  boost::int16_t get_DV( boost::uint8_t motor ) const;

  void set_Brake( boost::uint8_t motor, bool on );
  bool get_Brake( boost::uint8_t motor ) const;


  void set_DP( boost::uint8_t motor, boost::int32_t value );
  void set_MODE( boost::uint8_t motor, boost::uint8_t value );

  void set_KP( boost::uint8_t motor, boost::uint8_t value );
  boost::uint8_t get_KP( boost::uint8_t motor ) const;

  void set_KI( boost::uint8_t motor, boost::uint8_t value );
  boost::uint8_t get_KI( boost::uint8_t motor ) const;

  void set_KD( boost::uint8_t motor, boost::uint8_t value );
  boost::uint8_t get_KD( boost::uint8_t motor ) const;

  void set_Shutdown( bool value );

  /**ResetPosition wird von RobotinoView gesetzt und im Microcontroller ausgewertet. get Funktion nur zur Kompatibilität mit Robertino.*/
  void set_ResetPosition( unsigned int motor, bool reset );
  bool get_ResetPosition( unsigned int motor ) const;

  /**ResetMotorTime wird von RobotinoView gesetzt und im Microcontroller ausgewertet. get Funktion nur zur Kompatibilität mit Robertino.*/
  void set_ResetMotorTime( unsigned int motor, bool reset );
  bool get_ResetMotorTime( unsigned int motor ) const;
  
  void set_DI( const std::string& name, boost::uint8_t value );
  boost::uint8_t get_DI( const std::string& name ) const;

  int get_Error( boost::uint8_t motor ) const;

  boost::uint8_t get_MasterTime() const;

  /**MotorTime wird von den Microcontrollern gesetzt. set Funktion nur zur Kompatibilität mit Robertino.*/
  void set_MotorTime( boost::uint8_t motor, boost::uint32_t time );
  boost::uint32_t get_MotorTime( boost::uint8_t motor ) const;

  /**AV ( actual velocity ) wird von den Microcontrollern gesetzt. set Funktion nur zur Kompatibilität mit Robertino.*/
  void set_AV( boost::uint8_t motor, boost::int16_t value );
  boost::int16_t get_AV( boost::uint8_t motor ) const;

  /**AP ( actual position ) wird von den Microcontrollern gesetzt. set Funktion nur zur Kompatibilität mit Robertino.*/
  void set_AP( boost::uint8_t motor, boost::int32_t position );
  boost::int32_t get_AP( boost::uint8_t motor ) const;

  /**AD ( analog digital converter ) wird von den Microcontrollern gesetzt. set Funktion nur zur Kompatibilität mit Robertino.*/
  void set_AD( const std::string& name, boost::uint16_t value );
  boost::uint16_t get_AD( const std::string& name ) const;

  std::list< std::string > doMapKeys() const;
  std::list< std::string > diMapKeys() const;
  std::list< std::string > adiMapKeys() const;

  unsigned int get_FirmwareVersion() const;

private:
  void set_DV_i( boost::uint8_t motor, boost::int16_t value );
  void set_KP_i( boost::uint8_t motor, boost::uint8_t value );
  void set_KI_i( boost::uint8_t motor, boost::uint8_t value );
  void set_KD_i( boost::uint8_t motor, boost::uint8_t value );

	void init();

  static std::map< std::string, PortIdentifier > _doMap;
  static std::map< std::string, PortIdentifier > _diMap;
  static std::map< std::string, ADIdentifier > _adiMap;

  // buffer without start sequence
  unsigned char* _p2q_ws;
  unsigned char* _q2p_ws;
};

#endif


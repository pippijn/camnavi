//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/serialport/SerialPort.h"

#ifdef WIN32
  #include "rec/serialport/SerialPortImpl_win.h"
#else
  #include "rec/serialport/SerialPortImpl_linux.h"
#endif

using rec::serialport::Port;
using rec::serialport::SerialPort;
using rec::serialport::SerialPortImpl;

SerialPort::SerialPort()
{
#ifdef WIN32
  _impl = new rec::serialport::SerialPortImpl_win();
#else
  _impl = new rec::serialport::SerialPortImpl_Linux();
#endif
}

SerialPort::SerialPort( SerialPortImpl* impl )
: _impl( impl )
{
}

SerialPort::~SerialPort()
{
  delete _impl;
}

bool SerialPort::isOpen() const
{
  return _impl->isOpen();
}

void SerialPort::open( Port port, unsigned int speed, unsigned int readTimeout )
{
  _impl->open( port, speed, readTimeout );
}

void SerialPort::open( const std::string& port, unsigned int speed, unsigned int readTimeout )
{
  _impl->open( port, speed, readTimeout );
}

rec::serialport::Port SerialPort::port() const
{
	return _impl->port();
}

void SerialPort::close()
{
  _impl->close();
}

void SerialPort::setSpeed( unsigned int speed )
{
  _impl->setSpeed( speed );
}

unsigned int SerialPort::speed() const
{
	return _impl->speed();
}

void SerialPort::setReadTimeout( unsigned int timeout )
{
  _impl->setReadTimeout( timeout );
}

int SerialPort::read( unsigned char* buffer, unsigned int length )
{
  return _impl->read( buffer, length );
}

//rec::core_lt::memory::ByteArray SerialPort::read( unsigned int bytesToRead )
//{
//	rec::core_lt::memory::ByteArray ba( bytesToRead );
//
//	int bytesRead = read( ba.data(), ba.size() );
//
//	if( bytesRead != ba.size() ) //error
//	{
//		return rec::core_lt::memory::ByteArray();
//	}
//	else
//	{
//		return ba;
//	}
//}

int SerialPort::write( const unsigned char* buffer, unsigned int length )
{
  return _impl->write( buffer, length );
}

//int SerialPort::write( const rec::core_lt::memory::ByteArrayConst& ba )
//{
//	return write( ba.constData(), ba.size() );
//}

void SerialPort::flush()
{
  return _impl->flush();
}

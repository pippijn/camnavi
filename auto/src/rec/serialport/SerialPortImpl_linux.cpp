//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/serialport/SerialPortImpl_linux.h"

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>

#include <iostream>
#include <string>
#include <map>

#define INVALID_HANDLE_VALUE -1

using rec::serialport::Port;
using rec::serialport::SerialPortImpl_Linux;
using rec::serialport::SerialPortException;

SerialPortImpl_Linux::SerialPortImpl_Linux()
: _fd( INVALID_HANDLE_VALUE )
, _speed( 0 )
, _port( rec::serialport::UNDEFINED )
{
}

SerialPortImpl_Linux::~SerialPortImpl_Linux()
{
  close();
}

bool SerialPortImpl_Linux::isOpen() const
{
  return (_fd != INVALID_HANDLE_VALUE);
}

void SerialPortImpl_Linux::open( Port port, unsigned int speed, unsigned int readTimeout )
{
	_port = port;
	open( getPortString( port ), speed, readTimeout );
}

void SerialPortImpl_Linux::open( const std::string& port, unsigned int speed, unsigned int readTimeout )
{
	if( isOpen() )
		throw SerialPortException("Port is already open");

	_fd = ::open( port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY );

	if( isOpen() )
	{
		fcntl( _fd, F_SETFL, 0 );
		configure( speed, readTimeout );
	}
	else
	{
		throw SerialPortException("Couldn't open port");
	}
}

rec::serialport::Port SerialPortImpl_Linux::port() const
{
	return _port;
}

void SerialPortImpl_Linux::close()
{
  if( isOpen() )
  {
    ::close( _fd );
    _fd = INVALID_HANDLE_VALUE;
	_port = rec::serialport::UNDEFINED;
  }
}


void SerialPortImpl_Linux::setSpeed( unsigned int speed )
{
  struct termios options;
  //Get the current options for the port...
  tcgetattr( _fd, &options);
  //Set the baud rates to speed...
	_speed = speed;
  speed = convSpeed( speed );
  cfsetispeed( &options, speed );
  cfsetospeed( &options, speed );
  tcsetattr( _fd, TCSANOW, &options );
}

unsigned int SerialPortImpl_Linux::speed() const
{
	return _speed;
}

void SerialPortImpl_Linux::setReadTimeout( unsigned int timeout )
{
  struct termios options;
  //Get the current options for the port...
  tcgetattr( _fd, &options);
  options.c_cc[VTIME] = timeout / 100;
  tcsetattr( _fd, TCSANOW, &options );
}

int SerialPortImpl_Linux::read( unsigned char* buffer, unsigned int length )
{
	/*
	If MIN = 0 and TIME > 0, TIME serves as a timeout value. The read will be satisfied if a single character is read, or TIME is exceeded (t = TIME *0.1 s). If TIME is exceeded, no character will be returned. 
	If MIN > 0 and TIME > 0, TIME serves as an inter-character timer. The read will be satisfied if MIN characters are received, or the time between two characters exceeds TIME. The timer is restarted every time a character is received and only becomes active after the first character has been received. 
	*/

	int bytes_read = 0;
	while( bytes_read < length )
	{
		int res = ::read( _fd, buffer+bytes_read, length-bytes_read );
		if( res < 1 )
		{
			//std::cout << "read less than 1 character" << std::endl;
			break;
		}

		bytes_read += res;
		//std::cout << "read " << bytes_read << " of " << length << " characters" << std::endl;
	}
	return bytes_read;
}

int SerialPortImpl_Linux::write( const unsigned char* buffer, unsigned int length )
{
  return ::write( _fd, buffer, length );
}

void SerialPortImpl_Linux::flush()
{
  tcflush( _fd, TCIOFLUSH ); 
}

void SerialPortImpl_Linux::configure( unsigned int speed, unsigned int readTimeout )
{
  struct termios options;

  //Get the current options for the port...
  tcgetattr( _fd, &options);

  //Set the baud rates to 115200...

	_speed = speed;
  speed = convSpeed( speed );
  cfsetispeed( &options, speed );
  cfsetospeed( &options, speed );

  //Enable the receiver and set local mode...
  options.c_cflag |= CLOCAL;
  options.c_cflag |= CREAD;
  options.c_cflag &= ~CSIZE;
  options.c_cflag |= CS8;
  //options.c_cflag |=  CRTSCTS;
  options.c_cflag &= ~CRTSCTS;
  options.c_cflag &= ~CSTOPB;
  options.c_cflag &= ~PARENB;
  
  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

  options.c_iflag &= ~(IXON|IXOFF|IXANY); /* Bye to IXON/IXOFF */
  options.c_iflag &= ~INLCR; //Map NL to CR
  options.c_iflag &= ~ICRNL; //Map CR to NL

  options.c_oflag &= ~OPOST; //Choosing Raw Output

/*
  options.c_oflag &= ~ONLCR; //Map NL to CR-NL
  options.c_oflag &= ~OCRNL; //Map CR to NL
  options.c_oflag |= NL0; //No delay for NLs
  options.c_oflag |= CR0; //No delay for CRs
  options.c_oflag |= TAB0; //No delay for TABs
*/

  options.c_cc[VMIN] = 0;
  options.c_cc[VTIME] = readTimeout / 100;

  //Set the new options for the port...
  tcsetattr( _fd, TCSANOW, &options );
}

const char* SerialPortImpl_Linux::getPortString(Port port) const
{
  switch(port)
  {
  case COM1:
    return "/dev/ttyS0";
  case COM2:
    return "/dev/ttyS1";
  case COM3:
    return "/dev/ttyS2";
  case COM4:
    return "/dev/ttyS3";
  case USB0:
    return "/dev/ttyUSB0";
	case USB1:
    return "/dev/ttyUSB1";
	case USB2:
    return "/dev/ttyUSB2";
	case USB3:
    return "/dev/ttyUSB3";
	case USB4:
    return "/dev/ttyUSB4";
	case USB5:
    return "/dev/ttyUSB5";
	case USB6:
    return "/dev/ttyUSB6";
	case USB7:
    return "/dev/ttyUSB7";
  default:
    throw SerialPortException("Port not supported by this implementation");
  }
}

unsigned int SerialPortImpl_Linux::convSpeed( unsigned int speed ) const
{
  unsigned int ret = B115200;
  switch( speed )
  {
  case 1200:
    ret = B1200;
    break;
  case 4800:
    ret = B4800;
    break;
  case 9600:
    ret = B9600;
    break;
  case 19200:
    ret = B19200;
    break;
  case 38400:
    ret = B38400;
    break;
  case 57600:
    ret = B57600;
    break;
  case 115200:
    ret = B115200;
    break;
  }
  return ret;
}

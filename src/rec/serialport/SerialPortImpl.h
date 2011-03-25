//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_SERIALPORT_SERIALPORTIMPL_H_
#define _REC_SERIALPORT_SERIALPORTIMPL_H_

#include "rec/serialport/Port.h"
#include "rec/serialport/SerialPortException.h"

namespace rec
{
	namespace serialport
	{
		class SerialPortImpl
		{
		public:
			virtual void open( Port port, unsigned int speed, unsigned int readTimeout ) = 0;
			virtual void open( const std::string& port, unsigned int speed, unsigned int readTimeout ) = 0;
			virtual Port port() const = 0;
			virtual bool isOpen() const = 0;
			virtual void close() = 0;

			virtual void setSpeed( unsigned int speed ) = 0;
			virtual unsigned int speed() const = 0;
			virtual void setReadTimeout( unsigned int timeout ) = 0;

			virtual int read( unsigned char* buffer, unsigned int length ) = 0;
			virtual int write( const unsigned char* buffer, unsigned int length ) = 0;
			virtual void flush() = 0;
		};
	}
}

#endif

//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_SERIALPORT_H_
#define _REC_SERIALPORT_H_

#include "rec/serialport/Port.h"
#include "rec/serialport/SerialPortException.h"

namespace rec
{
	namespace serialport
	{
		class SerialPortImpl;

		class SerialPort
		{
		public:
			SerialPort();
			virtual ~SerialPort();

			bool isOpen() const;

			/**
			@param port
			@param speed
			@param readTimeout is in ms. Under Linux readTimeout must be bigger than 100
			@throws SerialPortException
			*/
			void open( Port port, unsigned int speed = 115200, unsigned int readTimeout = 200 );

			/**
			@param port "COM1" ... "dev/ttyUSB0"
			@param speed BAUD rate
			@param readTimeout is in ms. Under Linux readTimeout must be bigger than 100
			@throws SerialPortException
			*/
			void open( const std::string& port, unsigned int speed = 57600, unsigned int readTimeout = 200 );

			void close();

			Port port() const;

			// speed in bit/sec, e.g. 115200
			void setSpeed( unsigned int speed );

			unsigned int speed() const;

			// 0 makes read blocking, timeout is in ms
			void setReadTimeout( unsigned int timeout );

			// returns bytes read or -1 if error
			int read( unsigned char* buffer, unsigned int length );

			// returns bytes written or -1 if error
			int write( const unsigned char* buffer, unsigned int length );

			void flush();

			/*
			// meassure throughput
			// returns number of bytes read since last call. flush resets the counter
			unsigned int bytesReadUpdate();

			// returns number of bytes written since last call. flush resets the counter
			unsigned int bytesWrittenUpdate();
			*/

		protected:
			SerialPort( SerialPortImpl* impl );

			SerialPortImpl* _impl;

			/*
			// througput functions
			unsigned int _bytesRead;
			unsigned int _bytesWritten;
			*/
		};
	}
}

#endif

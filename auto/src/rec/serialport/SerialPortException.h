//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_SERIALPORT_SERIALPORTEXCEPTION_H_
#define _REC_SERIALPORT_SERIALPORTEXCEPTION_H_

#include "rec/core_lt/Exception.h"

namespace rec
{
	namespace serialport
	{
		class SerialPortException : public rec::core_lt::Exception
		{
		public:
			SerialPortException(const std::string &description) : rec::core_lt::Exception(description)
			{
			}
		};
	}
}

#endif

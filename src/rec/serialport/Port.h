//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_SERIALPORT_PORT_H_
#define _REC_SERIALPORT_PORT_H_

#include <string>

namespace rec
{
	namespace serialport
	{
		enum Port { UNDEFINED, COM1, COM2, COM3, COM4, COM5, COM6, COM7, COM8, COM9, COM10, COM11, COM12, USB0, USB1, USB2, USB3, USB4, USB5, USB6, USB7 };

		static std::string friendlyName( int port )
		{
			switch( port )
			{
			case COM1:
				return "COM1";

			case COM2:
				return "COM2";

			case COM3:
				return "COM3";

			case COM4:
				return "COM4";

			case COM5:
				return "COM5";

			case COM6:
				return "COM6";

			case COM7:
				return "COM7";

			case COM8:
				return "COM8";

			case COM9:
				return "COM9";

			case COM10:
				return "COM10";

			case COM11:
				return "COM11";

			case COM12:
				return "COM12";

			case USB0:
				return "USB0";

			case USB1:
				return "USB1";

			case USB2:
				return "USB2";

			case USB3:
				return "USB3";

			case USB4:
				return "USB4";

			case USB5:
				return "USB5";

			case USB6:
				return "USB6";

			case USB7:
				return "USB7";

			default:
				return "Undefined";
			}
		}
	}
}

#endif

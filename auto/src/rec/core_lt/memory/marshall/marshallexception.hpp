//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_MARSHALL_MARSHALLEXCEPTION_H_
#define _REC_MARSHALL_MARSHALLEXCEPTION_H_

#include "rec/core_lt/Exception.h"

#include <sstream>

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			namespace marshall
			{
				class MarshallException : public rec::core_lt::Exception
				{
				public:
					MarshallException(const std::string &description) : rec::core_lt::Exception(description)
					{
					}
				};
			}
		}
	}
}

#endif

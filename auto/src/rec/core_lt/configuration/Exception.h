//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_CONFIGURATION_EXCEPTION_H_
#define _REC_CORE_LT_CONFIGURATION_EXCEPTION_H_

#include "rec/core_lt/Exception.h"

#include <sstream>

namespace rec
{
	namespace core_lt
	{
		namespace configuration
		{
			class Exception : public rec::core_lt::Exception
			{
			public:
				static Exception createElementNoFoundException( const std::string configname, const std::string &element )
				{
					std::ostringstream os;
					os << "The configuration '" << configname << "' doesn't contain an element '" << element << "'";
					return Exception( os.str() );
				}

				static Exception createCannotConvertException( const std::string configname, const std::string &element )
				{
					std::ostringstream os;
					os << "The element '" << element << "' in the configuration '" + configname << "' cannot be converted to the requested type";;
					return Exception( os.str() );
				}

				Exception(const std::string &description) : rec::core_lt::Exception( description )
				{
				}
			};
		}
	}
}

#endif //_REC_CORE_LT_CONFIGURATION_EXCEPTION_H_


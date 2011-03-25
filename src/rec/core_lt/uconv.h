//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_UCONV_H_
#define _REC_UCONV_H_

#include "rec/core_lt/defines.h"

#include <string>

namespace rec
{
	namespace core_lt
	{
		REC_CORE_LT_EXPORT std::wstring toWString( const std::string& str );

		REC_CORE_LT_EXPORT std::string toStdString( const std::wstring& ws );
	}
}

#endif

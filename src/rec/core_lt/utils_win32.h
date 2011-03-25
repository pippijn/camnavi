//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifdef WIN32

#ifndef _REC_CORE_LT_UTILS_WIN32_H_
#define _REC_CORE_LT_UTILS_WIN32_H_

#include "rec/core_lt/defines.h"

#include <windows.h>
#include <string>

namespace rec
{
	namespace core_lt
	{
		REC_CORE_LT_EXPORT std::string fromWideChar( LPCWSTR wide );

		REC_CORE_LT_EXPORT std::string getShortPathName( LPCTSTR fileName );
	}
}

#endif //_REC_CORE_LT_UTILS_WIN32_H_

#endif //WIN32


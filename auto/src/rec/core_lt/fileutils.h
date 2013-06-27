//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_FILEUTILS_H_
#define _REC_FILEUTILS_H_

#include "rec/core_lt/defines.h"

#include <string>

namespace rec
{
	namespace core_lt
	{
		namespace fileutils
		{
			REC_CORE_LT_EXPORT std::string readLineFromFile( const char* filename );

			REC_CORE_LT_EXPORT bool isContainedIn( const std::string& baseDir, const std::string& relativeSubEntry );

			REC_CORE_LT_EXPORT std::string readFileToString( const std::string& filename );

			/// only works with unix-like directory separators for crazy filenames
			REC_CORE_LT_EXPORT std::string getExtension( const std::string& filename );
		}
	}
}

#endif

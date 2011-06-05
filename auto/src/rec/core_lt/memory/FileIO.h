//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_FILEIO_H_
#define _REC_CORE_LT_MEMORY_FILEIO_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/memory/ByteArray.h"
#include <string>

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			REC_CORE_LT_EXPORT ByteArray read( const std::string& filename );
			REC_CORE_LT_EXPORT bool write( const std::string& filename, const ByteArray& data );
		}
	}
}

#endif

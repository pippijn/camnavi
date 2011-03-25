//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_SHAREDMEMORY_SHAREDMEMORYEXCEPTION_H_
#define _REC_SHAREDMEMORY_SHAREDMEMORYEXCEPTION_H_

#include "rec/core_lt/Exception.h"

namespace rec
{
namespace sharedmemory
{
	class SharedMemoryException : public rec::core_lt::Exception
{
public:
  SharedMemoryException(const std::string &description) : rec::core_lt::Exception(description)
  {
  }
};
}
}

#endif

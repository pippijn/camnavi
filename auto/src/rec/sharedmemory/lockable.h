//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_SHAREDMEMORY_LOCKABLE_H_
#define _REC_SHAREDMEMORY_LOCKABLE_H_

namespace rec
{
namespace sharedmemory
{
class Lockable
{
public:
  virtual bool lock() = 0;
  virtual bool unlock() = 0;
};
}
}

#endif

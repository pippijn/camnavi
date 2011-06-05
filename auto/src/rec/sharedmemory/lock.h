//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_SHAREDMEMORY_LOCK_H_
#define _REC_SHAREDMEMORY_LOCK_H_

#include <rec/sharedmemory/lockable.h>

namespace rec
{
namespace sharedmemory
{
  class Lock
  {
  private:
    Lockable* lockable;

  public:
    Lock(Lockable *lockable)
    {
      this->lockable = lockable;
      lockable->lock();
    }

    virtual ~Lock()
    {
      lockable->unlock();
    }
  };
}
}

#endif

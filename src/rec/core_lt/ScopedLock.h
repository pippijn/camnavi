//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_SCOPEDLOCK_H_
#define _REC_CORE_LT_SCOPEDLOCK_H_

#include "rec/core_lt/Lockable.h"

namespace rec
{
  namespace core_lt
  {
    class ScopedLock
    {
    private:
      Lockable* lockable;

    public:
      ScopedLock( Lockable *lockable )
      {
        this->lockable = lockable;
        lockable->lock();
      }

      virtual ~ScopedLock()
      {
        lockable->unlock();
      }
    };
  }
}

#endif

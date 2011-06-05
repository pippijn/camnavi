//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_LOCKABLE_H_
#define _REC_CORE_LT_LOCKABLE_H_

namespace rec
{
  namespace core_lt
  {
    class Lockable
    {
    public:
      virtual void lock() = 0;
      virtual void unlock() = 0;
    };
  }
}

#endif

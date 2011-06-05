//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_SHAREDMEMORY_H_
#define _REC_CORE_LT_MEMORY_SHAREDMEMORY_H_

#include "rec/core_lt/Exception.h"
#include "rec/core_lt/Lockable.h"
#include "rec/core_lt/ScopedLock.h"


namespace rec
{
  namespace core_lt
  {
    namespace memory
    {
      class SharedMemoryImpl;

      template< typename sharedType >
      class SharedMemory : public Lockable
      {
      public:
        SharedMemory( int key );
        ~SharedMemory();
        void lock();
        void unlock();
        sharedType* getData() const;

      private:
        sharedType* data;
        int key;
        bool owner;
        SharedMemoryImpl* impl;
      };

#ifdef WIN32
#include "rec/core_lt/memory/SharedMemoryImpl_win.h"
#else
#include "rec/core_lt/memory/SharedMemoryImpl_linux.h"
#endif
    }
  }
}

#endif

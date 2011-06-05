//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_SHAREDMEMORY_SHAREDMEMORY_H_
#define _REC_SHAREDMEMORY_SHAREDMEMORY_H_

#include <rec/sharedmemory/sharedmemoryexception.h>
#include <rec/sharedmemory/lockable.h>
#include <rec/sharedmemory/lock.h>

#include <iostream>
#include <sstream>
#include <boost/cstdint.hpp>

namespace rec
{
namespace sharedmemory
{
class SharedMemoryImpl;

template< typename sharedType >
class SharedMemory : public Lockable
{
public:
  SharedMemory( int key );
  ~SharedMemory();
  bool lock();
  bool unlock();
  sharedType* getData();

private:
  sharedType* data;
  int key;
  bool owner;
  SharedMemoryImpl* impl;
};

#ifdef WIN32
  #include <rec/sharedmemory/sharedmemory_win.h>
#else
  #include <rec/sharedmemory/sharedmemory_linux.h>
#endif
}
}

#endif

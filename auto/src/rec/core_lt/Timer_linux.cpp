//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef WIN32

#include "rec/core_lt/Timer.h"
#include <sys/types.h>
#include <sys/time.h>

namespace rec
{
	namespace core_lt
	{
		class TimerImpl
		{
		public:
			TimerImpl();

			/**usecs elapsed between count1 and count2. Is positive when count2 > count1.*/
			static float timeDifference( const timeval& count1, const timeval& count2 );

			timeval _time;
		};

		TimerImpl::TimerImpl()
		{
		  _time.tv_sec = 0;
		}
		
		Timer::Timer()
		: _impl( new TimerImpl )
		{
		}
		
		Timer::~Timer()
		{
		  delete _impl;
		}
		
		float TimerImpl::timeDifference( const timeval& t1, const timeval& t2 )
		{
		  int usecdiff = t2.tv_usec - t1.tv_usec;
		  int secdiff = t2.tv_sec - t1.tv_sec;
		  if( usecdiff < 0 )
		  {
		    usecdiff += 1000000;
		    --secdiff;
		  }
		  return (float)((secdiff * 1000000) + usecdiff) / 1000.0f;
		}
		
		void Timer::start()
		{
		  gettimeofday( &_impl->_time, 0 );
		}
		
		float Timer::msecsElapsed() const
		{
		  if( ! isNull() )
		  {
		    timeval t;
		    gettimeofday( &t, 0 );
		    return TimerImpl::timeDifference( _impl->_time, t );
		  }
		  else
		  {
		    return 0.0f;
		  }
		}
		
		unsigned int Timer::frequency() const
		{
		  if( ! isNull() )
		  {
		    return (unsigned int)(1000.0f / msecsElapsed());
		  }
		  else
		  {
		    return 0;
		  }
		}
		
		bool Timer::isNull() const
		{
		  return _impl->_time.tv_sec == 0;
		}
		
		void Timer::reset()
		{
		  _impl->_time.tv_sec = 0;
		}
	}
}

#endif

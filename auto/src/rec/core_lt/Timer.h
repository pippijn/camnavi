//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_TIMER_H_
#define _REC_CORE_LT_TIMER_H_

#include "rec/core_lt/defines.h"

namespace rec
{
	namespace core_lt
	{
		class TimerImpl;

		class REC_CORE_LT_EXPORT Timer
		{
		public:
			Timer();
			~Timer();

			/**Start the timer. If the timer is already running, i.e. isNull() will return false, the timer is restarted.*/
			void start();

			/**Milliseconds since the last call to start(). Returns 0 if start() has not been called before.*/
			float msecsElapsed() const;

			/**1000 / msecsElapsed(). Returns 0 if start() has not been called before.*/
			unsigned int frequency() const;

			/**Resets the timer. isNull() returns true afterwards.*/
			void reset();

			/**Returns false if timing is started by a call to start(). Returns true before a call to start() or after
			calling reset().*/
			bool isNull() const;

		private:
			TimerImpl* _impl;
		};
	}
}

#endif //_REC_CORE_LT_TIMER_H_

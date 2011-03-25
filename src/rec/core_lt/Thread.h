//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_THREAD_H_
#define _REC_CORE_LT_THREAD_H_

#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/utility.hpp>

namespace rec
{
	namespace core_lt
	{
		class Thread : private boost::noncopyable
		{
		public:
			typedef boost::mutex::scoped_lock lock;

			Thread()
				: _run( false )
				,_active( false )
				,_thread( NULL )
			{
			}

			~Thread()
			{
				stop();
				if( NULL != _thread )
				{
					delete _thread;
				}
			}

			void start()
			{
				lock lk( _runMutex );
				if( ! _run && _active )
				{
					// thread cleanup not finished yet
					_stopCond.wait( lk );
				}
				if( ! _active )
				{
					if( NULL != _thread )
					{
						delete _thread;
					}
					_run = true;
					_active = true;
					_thread = new boost::thread( boost::bind( &Thread::run_i, this ) );
				}
			}

			virtual void stop()
			{
				lock lk( _runMutex );
				if( _active )
				{
					_run = false;
					_stopCond.wait( lk );
				}
			}

			void signalStop()
			{
				_run = false;
			}

			bool wait( unsigned int msTimeout )
			{
				lock lk( _runMutex );

				if( false == _active )
				{
					return true;
				}

				boost::xtime xt;
				boost::xtime_get( &xt, boost::TIME_UTC );
				xt.nsec += msTimeout * 1000000;
				xt.sec += xt.nsec / 1000000000;
				xt.nsec = xt.nsec % 1000000000;

				if( 0 == msTimeout )
				{
					_stopCond.wait( lk );
				}
				else
				{
					if( !_stopCond.timed_wait( lk, xt ) )
					{
						return false;
					}
				}

				return true;
			}

			bool isRunning() const
			{
				lock lk( _runMutex );
				return _active;
			}

		protected:
			void run_i()
			{
				run();
				lock lk( _runMutex );
				_run = false;
				_active = false;
				_stopCond.notify_all();
			}

			virtual void run() = 0;

			// cooperative child classes should check this variable
			bool _run;

			mutable boost::mutex _runMutex;

		private:
			// is true if thread is still active
			bool _active;
			boost::condition_variable _stopCond;
			boost::thread* _thread;
		};
	}
}

#endif

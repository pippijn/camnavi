//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_BLOCKINGQUEUE_H_
#define _REC_CORE_LT_MEMORY_BLOCKINGQUEUE_H_

#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <queue>
#include "rec/core_lt/Exception.h"
#include "rec/core_lt/utils.h"

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{

			class WaitAbortedException : public rec::core_lt::Exception
			{
			public:
				WaitAbortedException() : rec::core_lt::Exception( "Queue wait aborted" )
				{
				}
			};

			template < class T >
			class BlockingQueue : private boost::noncopyable
			{
			public:
				BlockingQueue()
					: _abortWaiting( false )
					,_numWaiting( 0 )
				{
				}

				~BlockingQueue()
				{
					abortWaiting();
				}

				void reset()
				{
					_queue.reset();
					_abortWaiting = false;
					_numWaiting = 0;
				}

				T dequeue()
				{
					boost::mutex::scoped_lock lock( _queueMutex );

					while( _queue.empty() )
					{
						++_numWaiting;
						_newItemCond.wait( lock );
						--_numWaiting;
						if( _abortWaiting )
						{
							throw WaitAbortedException();
						}
					}

					T item = _queue.front();
					_queue.pop();
					return item;
				}

				void enqueue( T item )
				{
					boost::mutex::scoped_lock lock( _queueMutex );
					_queue.push( item );
					_newItemCond.notify_one();
				}

				void abortWaiting()
				{
					while( _numWaiting != 0 )
					{
						_abortWaiting = true;
						_newItemCond.notify_all();
						rec::core_lt::msleep( 50 );
					}
				}

				bool isEmpty() const
				{
					boost::mutex::scoped_lock lock( _queueMutex );
					return _queue.empty();
				}

				bool contains( const T& item ) const
				{
					boost::mutex::scoped_lock lock( _queueMutex );
					std::queue< T >::const_iterator iter = _queue.begin();
					while( _queue.end() != iter )
					{
						if( item == *iter )
						{
							return true;
						}
						++iter;
					}
					return false;
				}

			private:
				bool _abortWaiting;
				mutable boost::mutex _queueMutex;
				unsigned int _numWaiting;
				boost::condition_variable _newItemCond;
				std::queue< T > _queue;
			};
		}
	}
}

#endif

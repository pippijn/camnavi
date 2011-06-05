//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_CYCLICBUFFER_H_
#define _REC_CORE_LT_MEMORY_CYCLICBUFFER_H_

#include "rec/core_lt/Exception.h"

#include <vector>

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/function.hpp>

#include <cassert>

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			template < class T > class ScopedWriteItem;
			template < class T > class ScopedReadItem;
			template < class T > class CyclicBuffer;

			template < class T > class CyclicBufferItemId
			{
				friend class CyclicBuffer<T>;
			public:
				CyclicBufferItemId()
					: _id( 0 )
				{
				}

				bool operator!=( const CyclicBufferItemId<T>& other ) const
				{
					return ( other._id != _id );
				}

				bool operator==( const CyclicBufferItemId<T>& other ) const
				{
					return ( other._id == _id );
				}

				CyclicBufferItemId<T>& operator=( const CyclicBufferItemId<T>& other )
				{
					_id = other._id;
					return *this;
				}

				unsigned int toUInt() const { return _id; }

			private:
				void operator++()
				{
					_id++;
				}

				unsigned int _id;
			};

			class CyclicBufferException : public Exception
			{
			public:
				CyclicBufferException( const std::string& message )
					: Exception( message )
				{
				}
			};

			template < class T > class CyclicBuffer
			{
				friend class ScopedWriteItem<T>;
				friend class ScopedReadItem<T>;
			public:
				CyclicBuffer(
					unsigned int size = 2,
					boost::function< T( unsigned int ) > ctor = boost::function< T( unsigned int ) >(),
					boost::function< void(T&) > dtor = boost::function< void(T&) >() )
					: _buffer( size )
					, _numLocks( size, 0 )
					, _writeIndex( 0 )
					, _readIndex( 0 )
					, _numWriters( 0 )
					, _dtor( dtor )
				{
					assert( size > 1 );

					resize( size, ctor );
				}

				~CyclicBuffer()
				{
					boost::mutex::scoped_lock lock( _mutex );
					cleanup();
				}

				unsigned int size() const { return _buffer.size(); }

				/// can only be called when no reader accesses 
				void resize( unsigned int s, boost::function< T( unsigned int ) > ctor = boost::function< T( unsigned int ) >() )
				{
					boost::mutex::scoped_lock lock( _mutex );

					unsigned int i;

					if( s == size() )
					{
						return;
					}
					if( _numWriters != 0 )
					{
						throw CyclicBufferException( "Writer still active" );
					}

					for( i = 0; i < size(); ++i )
					{
						if( _numLocks[ i ] > 0 )
						{
							throw CyclicBufferException( "Reader still active" );
						}
					}

					cleanup();

					// no resize because of constructor
					_buffer = std::vector< T >( s );
					_numLocks.resize( s );

					if( ctor )
					{
						for( unsigned int i=0; i<s; i++ )
						{
							_buffer[i] = ctor( i );
						}
					}
				}

			private:
				unsigned int getReadItem( CyclicBufferItemId<T>* oldId, unsigned int msTimeout ) const
				{
					boost::mutex::scoped_lock lock( _mutex );

					//std::cout << "old id " << oldId->toUInt() << "  current id " << _currentId.toUInt() <<  std::endl;

					if( _numWriters > 0 || ( ( NULL != oldId ) && ( *oldId == _currentId ) ) )
					{
						if( 0 == msTimeout )
						{
							_notifyWriteEnd.wait( lock );
						}
						else
						{
							boost::xtime xt;
							boost::xtime_get( &xt, boost::TIME_UTC );
							xt.nsec += msTimeout * 1000000;
							xt.sec += xt.nsec / 1000000000;
							xt.nsec = xt.nsec % 1000000000;

							if( !_notifyWriteEnd.timed_wait( lock, xt ) )
							{
								throw CyclicBufferException( "read timeout" );
							}

							//std::cout << "wait for write end finished" << std::endl;
						}
					}

					_numLocks[ _readIndex ]++;

					if( NULL != oldId )
					{
						*oldId = _currentId;
					}

					//std::cout << "going to read index " << _readIndex << std::endl;

					return _readIndex;
				}

				void releaseReadItem( unsigned int index ) const
				{
					boost::mutex::scoped_lock lock( _mutex );
					assert( _numLocks[ index ] > 0 );
					_numLocks[ index ]--;
					//std::cout << "release read item index: " << index << std::endl;
					if( 0 == _numLocks[ index ] )
					{
						_notifyReadEnd.notify_one();
					}
				}

				unsigned int getWriteItem( unsigned int msTimeout )
				{
					boost::mutex::scoped_lock lock( _mutex );

					boost::xtime xt;
					boost::xtime_get( &xt, boost::TIME_UTC );
					xt.nsec += msTimeout * 1000000;
					xt.sec += xt.nsec / 1000000000;
					xt.nsec = xt.nsec % 1000000000;

					while( _writeIndex == _readIndex && _numLocks[_readIndex] > 0 )
					{
						if( 0 == msTimeout )
						{
							_notifyReadEnd.wait( lock );
						}
						else
						{
							if( !_notifyReadEnd.timed_wait( lock, xt ) )
							{
								throw CyclicBufferException( "write timeout" );
							}
						}
					}

					unsigned int retIndex = _writeIndex;
					_writeIndex = ( _writeIndex + 1 ) % _buffer.size();
					++_numWriters;
					return retIndex;
				}

				void releaseWriteItem( unsigned int index )
				{
					boost::mutex::scoped_lock lock( _mutex );
					assert( _numWriters > 0 );
					_readIndex = index;
					++_currentId;
					--_numWriters;

					//std::cout << "relase write item  index: " << index << "   value: " << static_cast<int>( _buffer[index] ) << "   currentId: " << _currentId.toUInt() << std::endl;
					_notifyWriteEnd.notify_all();
				}

				void cleanup()
				{
					if( _dtor )
					{			
						for( unsigned int i=0; i<_buffer.size(); i++ )
						{
							_dtor( _buffer[i] );
						}
					}
				}

				std::vector<T> _buffer;
				mutable std::vector<unsigned int> _numLocks;

				unsigned int _writeIndex;
				unsigned int _readIndex;

				unsigned int _numWriters;

				CyclicBufferItemId<T> _currentId;

				mutable boost::mutex _mutex;
				mutable boost::condition_variable _notifyReadEnd;
				mutable boost::condition_variable _notifyWriteEnd;

				boost::function< void(T&) > _dtor;
			};

			template< class T > class ScopedReadItem : private boost::noncopyable
			{
			public:
				ScopedReadItem( const CyclicBuffer<T>& buffer, CyclicBufferItemId<T>* oldId = NULL, unsigned int msTimeout = 0 )
					: _buffer( buffer )
					, _readIndex( _buffer.getReadItem( oldId, msTimeout ) )
				{
				}

				~ScopedReadItem()
				{
					_buffer.releaseReadItem( _readIndex );
				}

				operator const T& () const
				{
					return _buffer._buffer[ _readIndex ];
				}

			private:
				const CyclicBuffer<T>& _buffer;
				unsigned int _readIndex;
			};

			template< class T > class ScopedWriteItem : private boost::noncopyable
			{
			public:
				ScopedWriteItem( CyclicBuffer<T>* buffer, unsigned int msTimeout = 0 )
					: _buffer( buffer )
					, _writeIndex( _buffer->getWriteItem( msTimeout ) )
				{
				}

				~ScopedWriteItem()
				{
					_buffer->releaseWriteItem( _writeIndex );
				}

				operator T& ()
				{
					return _buffer->_buffer[ _writeIndex ];
				}

				ScopedWriteItem& operator=( const T& data )
				{
					static_cast<T&>( *this ) = data;
					return *this;
				}

				unsigned int itemIndex() const { return _writeIndex; }

			private:
				CyclicBuffer<T>* _buffer;
				unsigned int _writeIndex;
			};
		}
	}
}

#endif //_REC_CORE_LT_MEMORY_CYCLICBUFFER_H_

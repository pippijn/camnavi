//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CACHE_H_
#define _REC_CACHE_H_

#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/function.hpp>
#include <list>
#include <set>
#include "rec/core_lt/Exception.h"

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			template < typename T >
			class Cache : private boost::noncopyable
			{
			public:
				Cache( unsigned int numStartEntries, T* copyItem )
					: _copyItem( copyItem )
				{
					for( unsigned int i = 0; i < numStartEntries; ++i )
					{
						_freeItems.push_back( new T( *copyItem ) );
					}
				}

				virtual ~Cache()
				{
					typename std::set< T* >::const_iterator ci = _aquiredItems.begin();
					while( ci != _aquiredItems.end() )
					{
						delete (*ci);
						ci++;
					}
					typename std::list< T* >::const_iterator ci2 = _freeItems.begin();
					while( ci2 != _freeItems.end() )
					{
						delete (*ci2);
						ci2++;
					}
					delete _copyItem;
				}

				T* aquireItem()
				{
					boost::mutex::scoped_lock lock( _itemsMutex );
					T* item;
					if( _freeItems.size() == 0 )
					{
						item = new T( *_copyItem );
					}
					else
					{
						item = _freeItems.front();
						_freeItems.pop_front();
					}
					_aquiredItems.insert( item );
					return item;
				}

				void releaseItem( T* item )
				{
					boost::mutex::scoped_lock lock( _itemsMutex );
					typename std::set< T* >::iterator iter = _aquiredItems.find( item );
					if( iter != _aquiredItems.end() )
					{
						_aquiredItems.erase( iter );
						_freeItems.push_front( item );
					}
					else
					{
						throw rec::core_lt::Exception( "Cache: item not found" );
					}
				}

			private:
				T* _copyItem;
				std::list< T* > _freeItems;
				std::set< T* > _aquiredItems;
				boost::mutex _itemsMutex;
			};
		}
	}
}

#endif

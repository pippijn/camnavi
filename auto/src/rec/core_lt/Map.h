//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MAP_H_
#define _REC_CORE_LT_MAP_H_

#include "rec/core_lt/SharedBase.h"

#include <map>

namespace rec
{
	namespace core_lt
	{
		template< typename Key, typename Value > class MapImpl
		{
		public:
			typedef std::map< Key, Value > map_t;
			
			map_t map;
		};

		template< typename Key, typename Value > class Map : rec::core_lt::SharedBase< MapImpl< Key, Value > >
		{
		private:
			typedef typename rec::core_lt::SharedBase< MapImpl< Key, Value > > BaseType;

		public:
			typedef typename MapImpl<Key,Value>::map_t::iterator iterator;
			typedef typename MapImpl<Key,Value>::map_t::const_iterator const_iterator;

			Map()
				: rec::core_lt::SharedBase< MapImpl< Key, Value > >( new MapImpl< Key, Value >() )
			{
			}

			iterator begin()
			{
				BaseType::detach();
				return BaseType::_impl->map.begin();
			}

			const_iterator begin() const
			{
				return BaseType::_impl->map.begin();
			}

			void clear()
			{
				BaseType::detach();
				BaseType::_impl->map.clear();
			}

			const_iterator constBegin() const
			{
				return begin();
			}

			const_iterator constEnd() const
			{
				return end();
			}

			const_iterator constFind( const Key& key ) const
			{
				return find( key );
			}

			bool contains( const Key& key ) const
			{
				return ( end() != find( key ) );
			}

			iterator end()
			{
				BaseType::detach();
				return BaseType::_impl->map.end();
			}

			const_iterator end() const
			{
				return BaseType::_impl->map.end();
			}

			iterator erase( iterator pos )
			{
				BaseType::detach();
				return BaseType::_impl->map.erase( pos );
			}

			iterator find( const Key& key )
			{
				BaseType::detach();
				return BaseType::_impl->map.find( key );
			}

			const_iterator find( const Key& key ) const
			{
				return BaseType::_impl->map.find( key );
			}

			bool isEmpty() const
			{
				return BaseType::_impl->map.empty();
			}

			const Key key( const Value& value ) const
			{
				return key( value, Key() );
			}

			const Key key( const Value& value, const Key& defaultKey ) const
			{
				const_iterator i = begin();
				while( end() != i )
				{
					if( value == (*i).second )
					{
						return (*i).first;
					}
					i++;
				}
				return defaultKey;
			}

			int remove( const Key& key )
			{
				BaseType::detach();
				int count = 0;
				iterator i = begin();
				while( end() != i )
				{
					if( key == (*i).first )
					{
						i = erase( i );
						count++;
					}
					else
					{
						i++;
					}
				}
				return count;
			}

			int size() const
			{
				return BaseType::_impl->map.size();
			}

			const Value value( const Key& key ) const
			{
				return value( key, Value() );
			}

			const Value value( const Key& key, const Value& defaultValue ) const
			{
				const_iterator i = find( key );
				if( end() != i )
				{
					return (*i).second;
				}
				return defaultValue;
			}

			bool operator!=( const Map<Key, Value>& other ) const
			{
				if( size() != other.size() )
				{
					return true;
				}

				const_iterator i = begin();
				const_iterator j = other.begin();
				while( end() != i && end() != j )
				{
					if( ( (*i).first != (*j).first ) || ( (*i).second != (*j).second ) )
					{
						return true;
					}
					i++;
					j++;
				}
				return false;
			}

			bool operator==( const Map<Key, Value>& other ) const
			{
				return ( !operator!=( other ) );
			}

			Value& operator[]( const Key & key )
			{
				BaseType::detach();
				return BaseType::_impl->map[ key ];
			}

			const Value operator[]( const Key& key ) const
			{
				return BaseType::_impl->map[ key ];
			}
		};
	}
}

#endif //_REC_CORE_LT_MAP_H_

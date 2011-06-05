//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_LIST_H_
#define _REC_CORE_LT_LIST_H_

#include "rec/core_lt/SharedBase.h"

#include <list>
#include <sstream>

namespace rec
{
	namespace core_lt
	{
		template< typename T > class Set;
		template< typename T > class Vector;

		template< typename T > class ListImpl
		{
		public:
			typedef std::list< T > list_t;
			
			ListImpl()
			{
			}

			ListImpl( unsigned int size )
				: list( size )
			{
			}

			template< typename InputIterator >
			ListImpl( InputIterator first, InputIterator last )
				: list( first, last )
			{
			}

			list_t list;
		};

		template< typename T > class List : rec::core_lt::SharedBase< ListImpl< T > >
		{
		private:
			typedef rec::core_lt::SharedBase< ListImpl< T > > BaseType;

		public:
			typedef T value_type;

			typedef typename ListImpl<T>::list_t::iterator iterator;
			typedef typename ListImpl<T>::list_t::const_iterator const_iterator;

			typedef typename ListImpl<T>::list_t::reference reference;
			typedef typename ListImpl<T>::list_t::const_reference const_reference;

			List()
				: rec::core_lt::SharedBase< ListImpl< T > >( new ListImpl< T >() )
			{
			}

			List( unsigned int size )
				: rec::core_lt::SharedBase< ListImpl< T > >( new ListImpl< T >( size ) )
			{
			}

			template< typename InputIterator >
			List( InputIterator first, InputIterator last )
				: rec::core_lt::SharedBase< ListImpl< T > >( new ListImpl< T >( first, last ) )
			{
			}

			virtual ~List()
			{
			}

			iterator append( const T& value )
			{
				BaseType::detach();
				BaseType::_impl->list.push_back( value );
				iterator iter = BaseType::_impl->list.end();
				--iter;
				return iter;
			}

			const_reference at( int i ) const
			{
				iterator iter = BaseType::_impl->list.begin();
				for( int j=0; j<i; ++j )
				{
					++iter;
				}
				//const_iterator iter = BaseType::_impl->list.begin() + i;
				return *iter;
			}

			iterator begin()
			{
				BaseType::detach();
				return BaseType::_impl->list.begin();
			}

			const_iterator begin() const
			{
				return BaseType::_impl->list.begin();
			}

			void clear()
			{
				BaseType::detach();
				BaseType::_impl->list.clear();
			}

			const_iterator constBegin() const
			{
				return begin();
			}

			const_iterator constEnd() const
			{
				return end();
			}

			bool contains( const T& value ) const
			{
				const_iterator i = std::find( begin(), end(), value );
				return ( end() != i );
			}

			int count( const T& value ) const
			{
				int count = 0;
				const_iterator i = begin();
				for( ;; )
				{
					i = std::find( i, end(), value );
					if( end() != i )
					{
						count++;
					}
					else
					{
						break;
					}
				}
				return count;
			}

			int count() const
			{
				return BaseType::_impl->list.size();
			}

			iterator end()
			{
				BaseType::detach();
				return BaseType::_impl->list.end();
			}

			const_iterator end() const
			{
				return BaseType::_impl->list.end();
			}

			iterator erase( iterator pos )
			{
				BaseType::detach();
				return BaseType::_impl->list.erase( pos );
			}

			iterator erase( iterator begin, iterator end )
			{
				BaseType::detach();
				return BaseType::_impl->list.erase( begin, end );
			}

			int indexOf( const T& value, int from = 0 ) const
			{
				assert( from >= 0 && from < size() );
				const_iterator iter = constBegin();

				for( int i=0; i<from; ++i )
				{
					++iter;
				}
				
				while( constEnd() != iter )
				{
					if( value == *iter )
					{
						return from;
					}
					iter++;
					from++;
				}
				return -1;
			}

			void insert( int index, const T& value )
			{
				assert( index >= 0 && index <= size() );
				iterator before = begin();
				for( int i=0; i<index; ++i )
				{
					++before;
				}
				insert( before, value );
			}

			void insert( iterator before, const T& value )
			{
				BaseType::detach();
				BaseType::_impl->list.insert( before, value );
			}

			bool isEmpty() const
			{
				return ( BaseType::_impl->list.empty() );
			}

			reference first()
			{
				BaseType::detach();
				return BaseType::_impl->list.front();
			}

			const_reference first() const
			{
				return BaseType::_impl->list.front();
			}

			reference last()
			{
				BaseType::detach();
				return BaseType::_impl->list.back();
			}

			const_reference last() const
			{
				return BaseType::_impl->list.back();
			}

			void prepend( const T& value )
			{
				BaseType::detach();
				BaseType::_impl->list.push_front( value );
			}

			void pop_back()
			{
				removeLast();
			}

			void pop_front()
			{
				removeFirst();
			}

			void push_back( const T& value )
			{
				append( value );
			}

			void push_front( const T& value )
			{
				prepend( value );
			}
			
			int removeAll( const T& value )
			{
				int count = 0;
				BaseType::detach();
				iterator i = begin();
				for( ;; )
				{
					i = std::find( i, end(), value );
					if( end() != i )
					{
						i = erase( i );
						count++;
					}
					else
					{
						break;
					}
				}
				return count;
			}

			void removeAt( int index )
			{
				assert( index >= 0 && index < size() );
				BaseType::detach();
				iterator iter = begin();
				for( int i = 0; i<index; ++i )
				{
					++iter;
				}
				BaseType::_impl->list.erase( iter );
			}

			void removeLast()
			{
				BaseType::detach();
				BaseType::_impl->list.pop_back();
			}

			void removeFirst()
			{
				BaseType::detach();
				BaseType::_impl->list.pop_front();
			}

			int size() const
			{
				return count();
			}

			void swap( List& other )
			{
				BaseType::detach();
				other.BaseType::detach();
				BaseType::_impl->list.swap( other._impl->list );
			}

			Set< T > toSet() const;

			Vector< T > toVector() const;

			T value( int i ) const
			{
				return value( i, T() );
			}

			T value( int i, const T& defaultValue ) const
			{
				if( i < 0 || i >= size() )
				{
					return defaultValue;
				}
				return at( i );
			}

			bool operator!= ( const List<T>& other ) const
			{
				if( size() != other.size() )
				{
					return true;
				}
				for( int i = 0; i<size(); i++ )
				{
					if( at(i) != other.at(i) )
					{
						return true;
					}
				}
				return false;
			}

			List<T>& operator<<( const T& value )
			{
				BaseType::detach();
				append( value );
				return *this;
			}

			List<T>& operator<<( const List<T>& other )
			{
				BaseType::detach();
				const_iterator iter = other.begin();
				while( other.end() != iter )
				{
					append( *iter );
					iter++;
				}
				return *this;
			}

			bool operator== ( const List<T>& other ) const
			{
				return ( !operator!=( other ) );
			}

			reference operator[] ( int index )
			{
				assert( index >= 0 && index < size() );

				BaseType::detach();
				iterator iter = BaseType::_impl->list.begin();
				for( int i=0; i<index; ++i )
				{
					++iter;
				}
				return *iter;
			}

			const_reference operator[] ( int i ) const
			{
				return at( i );
			}

			static List< T > fromSet( const rec::core_lt::Set< T >& set );

			static List< T > fromVector( const rec::core_lt::Vector< T >& vec );
		};

		template< class T > std::ostream& operator<<( std::ostream& os, const List< T >& l )
		{
			os << "(";
			typename rec::core_lt::List< T >::const_iterator i = l.constBegin();
			typename rec::core_lt::List< T >::const_iterator last = l.constEnd();
			if ( i != last )
			{
				--last;
			}
			while( l.constEnd() != i )
			{
				os << *i;

				if( i != last )
				{
					os << " ";
				}

				++i;
			}
			os << ")";
			return os;
		}

		template< class T > std::istream& operator>>( std::istream& is, List< T >& l )
		{
			char ch;
			is >> ch;
			if( '(' != ch )
			{
				is.setstate( std::ios_base::failbit );
				return is;
			}

			while( is.good() )
			{
				is >> ch;
				if ( ch == ')' )
					break;
				is.unget();
				T value;
				is >> value;

				l.append( value );
			}
			return is;
		}
	}
}

#ifdef HAVE_QT
#include <QtDebug>
template< typename T >
QDebug operator<<( QDebug dbg, const rec::core_lt::List< T >& l )
{
	dbg << "(";
	typename rec::core_lt::List< T >::const_iterator i = l.constBegin();
	while( l.constEnd() != i )
	{
		dbg << *i;
		++i;
	}
	dbg << ")";
	return dbg;
}
#endif

#endif // _REC_CORE_LIST_H_

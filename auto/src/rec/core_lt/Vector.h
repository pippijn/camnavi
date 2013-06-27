//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_VECTOR_H_
#define _REC_CORE_LT_VECTOR_H_

#include "rec/core_lt/SharedBase.h"

#include <vector>
#include <sstream>

namespace rec
{
	namespace core_lt
	{
		template< typename T > class List;
		template< typename T > class Set;

		template< typename T > class VectorImpl
		{
		public:
			typedef std::vector< T > vector_t;
			
			VectorImpl()
			{
			}

			VectorImpl( unsigned int size )
				: vec( size )
			{
			}

			VectorImpl( unsigned int size, const T& value )
				: vec( size )
			{
				typename vector_t::iterator i = vec.begin();
				while( vec.end() != i )
				{
					*i = value;
					++i;
				}
			}

			template< typename InputIterator >
			VectorImpl( InputIterator first, InputIterator last )
				: vec( first, last )
			{
			}

			vector_t vec;
		};

		template< typename T > class Vector : rec::core_lt::SharedBase< VectorImpl< T > >
		{
		private:
			typedef rec::core_lt::SharedBase< VectorImpl< T > > BaseType;

		public:
			typedef T value_type;

			typedef typename VectorImpl<T>::vector_t::iterator iterator;
			typedef typename VectorImpl<T>::vector_t::const_iterator const_iterator;

			typedef typename VectorImpl<T>::vector_t::reference reference;
			typedef typename VectorImpl<T>::vector_t::const_reference const_reference;

			Vector()
				: rec::core_lt::SharedBase< VectorImpl< T > >( new VectorImpl< T >() )
			{
			}

			Vector( unsigned int size )
				: rec::core_lt::SharedBase< VectorImpl< T > >( new VectorImpl< T >( size ) )
			{
			}

			Vector( unsigned int size, const T& value )
				: rec::core_lt::SharedBase< VectorImpl< T > >( new VectorImpl< T >( size, value ) )
			{
			}

			template< typename InputIterator >
			Vector( InputIterator first, InputIterator last )
				: rec::core_lt::SharedBase< VectorImpl< T > >( new VectorImpl< T >( first, last ) )
			{
			}

			virtual ~Vector()
			{
			}

			void append( const T& value )
			{
				BaseType::detach();
				BaseType::_impl->vec.push_back( value );
			}

			const_reference at( int i ) const
			{
				return BaseType::_impl->vec[i];
			}

			const_reference at( unsigned int i ) const
			{
				return BaseType::_impl->vec[i];
			}

			iterator begin()
			{
				BaseType::detach();
				return BaseType::_impl->vec.begin();
			}

			const_iterator begin() const
			{
				return BaseType::_impl->vec.begin();
			}

			void clear()
			{
				BaseType::detach();
				BaseType::_impl->vec.clear();
			}

			const_iterator constBegin() const
			{
				return begin();
			}

			// This method cannot be used in a Vector<bool> because of incompatible internal data storage!!!
			const T* constData() const
			{
				return &BaseType::_impl->vec[0];
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
				return BaseType::_impl->vec.size();
			}

			// This method cannot be used in a Vector<bool> because of incompatible internal data storage!!!
			T* data()
			{
				BaseType::detach();
				return &BaseType::_impl->vec[0];
			}

			iterator end()
			{
				BaseType::detach();
				return BaseType::_impl->vec.end();
			}

			const_iterator end() const
			{
				return BaseType::_impl->vec.end();
			}

			iterator erase( iterator pos )
			{
				BaseType::detach();
				return BaseType::_impl->vec.erase( pos );
			}

			reference first()
			{
				BaseType::detach();
				iterator i = begin();
				return (*i);
			}

			const_reference first() const
			{
				const_iterator i = begin();
				return (*i);
			}

			int indexOf( const T& value, int from = 0 ) const
			{
				assert( from >= 0 );

				for( int i=from; i<size(); i++ )
				{
					if( value == at( i ) )
					{
						return i;
					}
				}
				return -1;
			}

			void insert( int i, const T& value )
			{
				if( i <= 0 )
				{
					prepend( value );
				}
				else if( i >= size() )
				{
					append( value );
				}
				else
				{
					BaseType::detach();
					BaseType::_impl->vec.insert( begin()+i, value );
				}
			}

			bool isEmpty() const
			{
				return ( BaseType::_impl->vec.empty() );
			}

			reference last()
			{
				BaseType::detach();
				iterator i = end();
				i--;
				return (*i);
			}

			const_reference last() const
			{
				const_iterator i = end();
				i--;
				return (*i);
			}

			void prepend( const T& value )
			{
				BaseType::detach();
				BaseType::_impl->vec.insert( begin(), value );
			}

			void pop_back()
			{
				BaseType::detach();
				BaseType::_impl->vec.pop_back();
			}

			void pop_front()
			{
				if ( size() > 0 )
				{
					BaseType::detach();
					BaseType::_impl->vec.erase( begin() );
				}
			}

			void push_back( const T& value )
			{
				append( value );
			}

			void push_front( const T& value )
			{
				prepend( value );
			}

			void remove( int i )
			{
				BaseType::detach();
				iterator iter = begin() + i;
				BaseType::_impl->vec.erase( iter );
			}

			void removeAll( const T& value )
			{
				BaseType::detach();
				iterator iter = begin();
				while( end() != iter )
				{
					if( *iter == value )
					{
						iter = BaseType::_impl->vec.erase( iter );
					}
					else
					{
						++iter;
					}
				}
			}

			void resize( int size, T c = T() )
			{
				if( size >= 0 )
				{
					BaseType::detach();
					BaseType::_impl->vec.resize( size, c );
				}
			}

			int size() const
			{
				return count();
			}

			void swap( Vector& other )
			{
				BaseType::detach();
				other.BaseType::detach();
				BaseType::_impl->vec.swap( other._impl->vec );
			}

			List< T > toList() const;

			Set< T > toSet() const;

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

			std::vector< T > toStdVector() const
			{
				return BaseType::_impl->vec;
			}

			static Vector<T> fromStdVector( const std::vector<T>& vec )
			{
				Vector<T> v;
				v._impl->vec = vec;
				return v;
			}

			bool operator!= ( const Vector<T>& other ) const
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

			Vector<T>& operator<<( const T& value )
			{
				BaseType::detach();
				append( value );
				return *this;
			}

			Vector<T>& operator<<( const Vector<T>& other )
			{
				BaseType::detach();
				for( int i=0; i<other.size(); i++ )
				{
					append( other.at( i ) );
				}
				return *this;
			}

			bool operator== ( const Vector<T>& other ) const
			{
				return ( !operator!=( other ) );
			}

			reference operator[] ( int i )
			{
				BaseType::detach();
				return BaseType::_impl->vec[i];
			}

			const_reference operator[] ( int i ) const
			{
				return BaseType::_impl->vec[i];
			}

			static Vector< T > fromList( const List< T >& list );

			static Vector< T > fromSet( const Set< T >& set );
		};

		template< class T > std::ostream& operator<<( std::ostream& os, const Vector< T >& v )
		{
			os << "(";
			for( int i=0; i<v.size(); ++i )
			{
				os << v[i];

				if( i < v.size()-1 )
				{
					os << " ";
				}
			}
			os << ")";
			return os;
		}

		template< class T > std::istream& operator>>( std::istream& is, Vector< T >& v )
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

				v.append( value );
			}
			return is;
		}

		typedef rec::core_lt::Vector< float > FloatVector;
		typedef rec::core_lt::Vector< double > DoubleVector;
		typedef rec::core_lt::Vector< bool > BoolVector;
		typedef rec::core_lt::Vector< int > IntVector;
		typedef rec::core_lt::Vector< unsigned int > UIntVector;
	}
}

#ifdef HAVE_QT
#include <QtDebug>
template< typename T >
QDebug operator<<( QDebug dbg, const rec::core_lt::Vector< T >& v )
{
	dbg << "(";
	for( int i = 0; i < v.size(); ++i )
	{
		dbg << v[ i ];
	}
	dbg << ")";
	return dbg;
}

#endif

#endif // _REC_CORE_LT_VECTOR_H_

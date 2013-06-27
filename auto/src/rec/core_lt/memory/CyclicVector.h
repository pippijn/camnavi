//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_CYCLICVECTOR_H_
#define _REC_CORE_LT_MEMORY_CYCLICVECTOR_H_

#include "rec/core_lt/Vector.h"

#include <cassert>

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			template < class T > class CyclicVector
			{
			public:
				template < class U > class Iterator
				{
					friend class CyclicVector<U>;
				public:
					bool operator!=( const Iterator& other ) const
					{
						return ( other._currentReadIndex != _currentReadIndex || other._overVector != _overVector );
					}

					bool operator==( const Iterator& other ) const
					{
						return ( other._currentReadIndex == _currentReadIndex && other._overVector == _overVector );
					}

					void operator++()
					{
						++_currentReadIndex;
						if( _currentReadIndex == _overVector->_buffer.size() )
						{
							_currentReadIndex = 0;
						}
						if( _startIndex == _currentReadIndex )
						{
							_currentReadIndex = _overVector->_buffer.size();
						}
					}

					const T& operator*()
					{
						return _overVector->_buffer.at( _currentReadIndex );
					}

				private:
					Iterator( const CyclicVector<U>* vector, const unsigned int currentReadIndex )
						: _overVector( vector )
						, _currentReadIndex( currentReadIndex )
						, _startIndex( currentReadIndex )
					{
					}

					const CyclicVector<U>* _overVector;
					unsigned int _currentReadIndex;
					const unsigned int _startIndex;
				};

				typedef Iterator<T> const_iterator;

				CyclicVector( unsigned int size )
					: _buffer( size )
					, _currentWriteIndex( 0 )
				{
				}

				CyclicVector( unsigned int size, const T& value )
					: _buffer( size, value )
					, _currentWriteIndex( 0 )
				{
				}

				void append( const T& value )
				{
					assert( _currentWriteIndex < _buffer.size() );
					_buffer[_currentWriteIndex] = value;
					++_currentWriteIndex;
					if( _currentWriteIndex == _buffer.size() )
					{
						_currentWriteIndex = 0;
					}
				}

				const_iterator begin() const
				{
					return Iterator<T>( this, _currentWriteIndex );
				}

				const_iterator constBegin() const
				{
					return begin();
				}

				const_iterator constEnd() const
				{
					return end();
				}

				const_iterator end() const
				{
					return Iterator<T>( this, _buffer.size() );
				}

			private:
				Vector<T> _buffer;
				unsigned int _currentWriteIndex;
			};
		}
	}
}

#endif

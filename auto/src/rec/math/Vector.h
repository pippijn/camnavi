#ifndef _REC_MATH_VECTOR_H_
#define _REC_MATH_VECTOR_H_

#include "rec/math/defines.h"
#include <sstream>
#include <cassert>

namespace rec
{
	namespace math
	{
		template< typename T, std::size_t Sz > class Vector
		{
		public:
			typedef T value_type;
			typedef T& reference;
			typedef const T& const_reference;
			typedef T* pointer;

			enum { Size = Sz }; //the size of the vector

			Vector()
#ifdef REC_MATH_DYNAMIC_MEMORY
				: _data( new value_type[ Size ] )
#endif
			{
				for( std::size_t i = 0; i < Size; i++ )
					_data[ i ] = value_type();
			}

			Vector( value_type x0, value_type x1 )
#ifdef REC_MATH_DYNAMIC_MEMORY
				: _data( new value_type[ Size ] )
#endif
			{
				assert( 2 == Size );
				_data[0] = x0;
				_data[1] = x1;
			}

			Vector( value_type x0, value_type x1, value_type x2 )
#ifdef REC_MATH_DYNAMIC_MEMORY
				: _data( new value_type[ Size ] )
#endif
			{
				assert( 3 == Size );
				_data[0] = x0;
				_data[1] = x1;
				_data[1] = x2;
			}

			Vector( const Vector& other )
#ifdef REC_MATH_DYNAMIC_MEMORY
				: _data( new value_type[ Size ] )
#endif
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] = other._data[i];
				}
			}

			Vector& operator=( const Vector& other )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] = other._data[i];
				}
				return *this;
			}

			virtual ~Vector()
			{
#ifdef REC_MATH_DYNAMIC_MEMORY
				delete [] _data;
#endif
			}

			value_type operator()( std::size_t i ) const
			{
				assert( i < Size );
				return _data[i];
			}

			reference operator()( std::size_t i )
			{
				assert( i < Size );
				return _data[i];
			}

			value_type operator[]( std::size_t i ) const
			{
				return this->operator()(i);
			}

			reference operator[]( std::size_t i )
			{
				return this->operator()(i);
			}

			Vector& operator+=( const Vector& other )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] += other._data[i];
				}
				return *this;
			}

			Vector& operator-=( const Vector& other )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] -= other._data[i];
				}
				return *this;
			}

			Vector& operator+=( value_type a )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] += a;
				}
				return *this;
			}

			Vector& operator-=( value_type a )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] -= a;
				}
				return *this;
			}

			Vector& operator*=( value_type a )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] *= a;
				}
				return *this;
			}

			Vector& operator/=( value_type a )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] /= a;
				}
				return *this;
			}

			Vector& operator%=( std::size_t a )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] %= a;
				}
				return *this;
			}

			Vector& operator^=( std::size_t a )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] ^= a;
				}
				return *this;
			}

			Vector& operator&=( std::size_t a )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] &= a;
				}
				return *this;
			}

			Vector& operator|=( std::size_t a )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] |= a;
				}
				return *this;
			}

			Vector& operator<<=( std::size_t a )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] << a;
				}
				return *this;
			}

			Vector& operator>>=( std::size_t a )
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					_data[i] >>= a;
				}
				return *this;
			}

			bool operator==( const Vector& other ) const
			{
				for( std::size_t i=0; i<Size; ++i )
				{
					if( fabs( _data[i] - other._data[i] ) > RealEpsilon )
					{
						return false;
					}
				}
				return true;
			}

			bool operator!=( const Vector& other ) const
			{
				return !operator==( other );
			}

			template<class U, std::size_t Len> friend Vector<U,Len> operator+( const Vector<U,Len>& a, const Vector<U,Len>& b );
			template<class U, std::size_t Len> friend Vector<U,Len> operator-( const Vector<U,Len>& a, const Vector<U,Len>& b );

			//vector product for Size==3 vectors
			template<class U, std::size_t Len> friend Vector<U,Len> operator*( const Vector<U,Len>& a, const Vector<U,Len>& b );

			template<class U, std::size_t Len> friend Vector<U,Len> operator*( const Vector<U,Len>& v, U a );
			template<class U, std::size_t Len> friend Vector<U,Len> operator/( const Vector<U,Len>& v, U a );

			// TODO: When this is enabled, instantiating operator<< or operator>> will not compile because of
			//       an ambiguity. However, as neither operator<< not operator>> needs access to private members
			//       of Vector, we can leave this commented out. Note that the other operators might suffer the
			//       same problems when they are actually instantiated!
			//template<class U, std::size_t Len> friend std::ostream& operator<<( std::ostream& os, const Vector<U,Len>& v );
			//template<class U, std::size_t Len> friend std::istream& operator>>( std::istream& os, Vector<U,Len>& v );

		private:
#ifdef REC_MATH_DYNAMIC_MEMORY
			pointer _data;
#else
			value_type _data[ Size ];
#endif
		};

		template< class U, std::size_t Len > std::ostream& operator<<( std::ostream& os, const Vector< U, Len >& v )
		{
			os << "(";
			for( std::size_t i=0; i<Len; ++i )
			{
				if( fabs( v[i] ) < RealEpsilon )
				{
					os << "0";
				}
				else
				{
					os << v[i];
				}
				if( i < Len-1 )
				{
					os << " ";
				}
			}
			os << ")";
			return os;
		}

		template< class U, std::size_t Len > std::istream& operator>>( std::istream& is, Vector< U, Len >& v )
		{
			char ch;
			is >> ch;
			if( '(' != ch )
			{
				is.setstate( std::ios_base::failbit );
				return is;
			}

			for( std::size_t i=0; i<Len; ++i )
			{
				if( is.good() )
				{
					is >> v[i];
				}
				else
				{
					v[i] = 0;
				}
			}
			if ( is.good() )
			{
				is >> ch; // closing bracket should be extracted from the stream
				// ch == ')'
			}
			return is;
		}

		template< class U, std::size_t Len > Vector< U, Len > operator+( const Vector< U, Len >& a, const Vector< U, Len >& b )
		{
			Vector< U, Len > ret;
			for( std::size_t i=0; i<ret.Size; ++i )
			{
				ret._data[i] = a._data[i] + b._data[i];
			}
			return ret;
		}

		template< class U, std::size_t Len > Vector< U, Len > operator-( const Vector< U, Len >& a, const Vector< U, Len >& b )
		{
			Vector< U, Len > ret;
			for( std::size_t i=0; i<ret.Size; ++i )
			{
				ret._data[i] = a._data[i] - b._data[i];
			}
			return ret;
		}

		template< class U, std::size_t Len > Vector< U, Len > operator*( const Vector< U, Len >& a, const Vector< U, Len >& b )
		{
			assert( 3 == Len );
			Vector< U, Len > ret( a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] );
			return ret;
		}

		template< class U, std::size_t Len > Vector< U, Len > operator*( const Vector< U, Len >& v, U a )
		{
			Vector< U, Len > ret( v );
			ret *= a;
			return ret;
		}

		template< class U, std::size_t Len > Vector< U, Len > operator*( U a, const Vector< U, Len >& v )
		{
			return v * a;
		}

		template< class U, std::size_t Len > Vector< U, Len > operator/( const Vector< U, Len >& v, U a )
		{
			Vector< U, Len > ret( v );
			ret /= a;
			return ret;
		}

		template< class U, std::size_t Len > Real dot( const Vector< U, Len >& a, const Vector< U, Len >& b )
		{
			Real ret = 0;
			for( std::size_t i=0; i<Len; ++i )
			{
				ret += a[i] * b[i];
			}
			return ret;
		}

		template< class U, std::size_t Len > Real sqrNorm2( const Vector< U, Len >& a )
		{
			return dot< U,Len >( a, a );
		}

		template< class U, std::size_t Len > Real norm2( const Vector< U, Len >& a )
		{
			return sqrt( sqrNorm2( a ) );
		}

		template< class U, std::size_t Len > Vector< U, Len > normalize( const Vector< U, Len >& a )
		{
			Vector< U, Len > ret( a );
			ret /= norm2( ret );
			return ret;
		}
	}
}

#ifdef HAVE_QT
#include <QtDebug>
template< class U, std::size_t Len > QDebug operator<<( QDebug dbg, const rec::math::Vector< U, Len >& v )
{
	dbg.nospace() << "(";
	for( std::size_t i = 0; i < Len; ++i )
	{
		if( fabs( v[i] ) < rec::math::RealEpsilon )
		{
			dbg << "0";
		}
		else
		{
			dbg << v[i];
		}
		if( i < Len-1 )
		{
			dbg << " ";
		}
	}
	dbg << ")";
	return dbg.space();
}

#endif
#endif

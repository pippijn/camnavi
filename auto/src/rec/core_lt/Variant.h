//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORELT_VARIANT_VARIANT_H_
#define _REC_CORELT_VARIANT_VARIANT_H_

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/any.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <map>

#include "rec/core_lt/defines.h"

#include "rec/core_lt/VariantId.h"

#ifdef WIN32
#include <typeinfo.h>
#endif

namespace rec
{
	namespace core_lt
	{
		/******************************************************************************/
		/***   Implementation of Variant template                                   ***/
		/******************************************************************************/
		/** The Variant template a data type that can store (and convert from and to)
		various data-types */
		class REC_CORE_LT_EXPORT Variant
		{
		public:
			// Constructors / Destructors //
			/// Default constructor (empty variant)
			Variant();

			/// Copy constructor
			Variant( const Variant& other );

			Variant& operator=( const Variant& other );

			/// Creates a variant of a specific type with specific value
			/// @param New variant's value
			template<typename ValueType>
			Variant( const ValueType & value )
				: _data( new boost::any( value ) )
			{
			}

			// Methods //
			/// @return iff variant is empty
			bool isEmpty() const;

			/// @return variant's type (void iff data not set)
			const std::type_info& type() const;

			/// @return variant's value
			template<typename T> T get( bool* ok = NULL ) const
			{
				if( NULL != ok ) *ok = true;
				if( 0 == _data.get() )
				{
					if( NULL != ok ) *ok = false;
					return T();
				}
				try
				{
					return boost::any_cast<T>( *_data.get() );
				}
				catch( const std::exception& )
				{
					if( NULL != ok ) *ok = false;
					return T();
				}
			}

			/// Equality
			bool operator==( const Variant& other ) const{ return ( other._id == _id ); }

			/// Inequality
			bool operator!=( const Variant& other ) const{ return ( other._id != _id ); }

			// Conversion methods


			/// @return Convert variant to int
			int toInt( bool* ok = NULL ) const;

			/// @return Convert variant to unsigned int
			unsigned int toUInt( bool* ok = NULL ) const;

			/// @return Convert variant to float
			float toFloat( bool* ok = NULL ) const;

			/// @return Convert variant to std::string
			std::string toString( bool* ok = NULL ) const;

			/// Create variant from int
			/// @param value Initialization value for new variant
			static Variant fromInt( int value )
			{
				return Variant( value );
			}

			/// Create variant from unsigned int
			/// @param value Initialization value for new variant
			static Variant fromUInt( unsigned int value )
			{
				return Variant( value );
			}

			/// Create variant from float
			/// @param value Initialization value for new variant
			static Variant fromFloat( float value )
			{
				return Variant( value );
			}

			/// Create variant from std::string
			/// @param value Initialization value for new variant
			static Variant fromString( const std::string& value )
			{
				return Variant( value );
			}

			VariantId id() const { return _id; }

		private:
			// Member variables //
			/// Container for data stored in variant
			boost::shared_ptr< boost::any > _data;
			VariantId _id;
		};
	}
}

#ifdef HAVE_QT
#include <QMetaType>
Q_DECLARE_METATYPE( rec::core_lt::Variant )
#endif

#endif // _REC_CORELT_VARIANT_VARIANT_H_

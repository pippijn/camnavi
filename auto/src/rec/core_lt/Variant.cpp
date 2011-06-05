//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/core_lt/Variant.h"

using namespace rec::core_lt;

Variant::Variant()
: _id( VariantId::null )
{
}

Variant::Variant( const Variant& other )
: _data( other._data )
, _id( other._id )
{
}

Variant& Variant::operator=( const Variant& other )
{
	_data = other._data;
	_id = other._id;
	return *this;
}

bool Variant::isEmpty() const
{
	return _id.isNull();
}

const std::type_info& Variant::type() const
{
	if( 0 == _data.get() )
	{
		return typeid(void);
	}
	return _data->type();
}

int Variant::toInt( bool* ok ) const
{
	return get<int>( ok );
}

unsigned int Variant::toUInt( bool* ok ) const
{
	return get<unsigned int>( ok );
}

float Variant::toFloat( bool* ok ) const
{
	return get<float>( ok );
}

std::string Variant::toString( bool* ok ) const
{
	return get<std::string>( ok );
}

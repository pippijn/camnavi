//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORELT_VARIANTID_H
#define _REC_CORELT_VARIANTID_H

#include "rec/core_lt/defines.h"

namespace rec
{
	namespace core_lt
	{
		class REC_CORE_LT_EXPORT VariantId
		{
		public:
			VariantId()
				: _id( g_id++ )
			{
			}

			bool operator==( const VariantId& other ) const { return other._id == _id; }
			bool operator!=( const VariantId& other ) const { return other._id != _id; }

			bool isNull() const { return ( null._id == _id ); }

			static VariantId null;

		private:
			VariantId( unsigned int i )
				: _id( i )
			{
			}

			static unsigned int g_id;

			unsigned int _id;
		};
	}
}

#ifdef HAVE_QT
#include <QMetaType>
Q_DECLARE_METATYPE(rec::core_lt::VariantId)
#endif

#endif //_REC_CORE_VARIANT_ID_H

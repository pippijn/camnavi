//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_STRINGLIST_H
#define _REC_CORE_LT_STRINGLIST_H

#include "rec/core_lt/defines.h"
#include "rec/core_lt/List.h"
#include "rec/core_lt/String.h"

#include <string>

namespace rec
{
	namespace core_lt
	{
		typedef List< String > StringList;

		REC_CORE_LT_EXPORT StringList split( const String& source, const String& separator );
		REC_CORE_LT_EXPORT StringList stringListFromString( const String& source );
		REC_CORE_LT_EXPORT String toString( const StringList& source );
	}
}

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(rec::core_lt::StringList)
#endif

#endif //_REC_CORE_LT_STRINGLIST_H

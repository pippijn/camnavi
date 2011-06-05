#include "rec/robotino/com/v1/messages/Info.h"

using namespace rec::robotino::com::v1::messages;

Info::Info( const QByteArray& data )
{
	for( int i = 0; i<2000; i++ )
	{
		_text += data.at( i );
	}

	_isPassiveMode = ( data.at( 2000 ) > 0 );
}


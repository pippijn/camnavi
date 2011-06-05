#include "rec/robotino/com/v1/messages/CameraControl.h"

using namespace rec::robotino::com::v1::messages;

QByteArray CameraControl::encode( unsigned int width, unsigned int height )
{
	QByteArray data( 8, 0 );

	data[0] = 2;
	data[1] = 5;
	data[2] = 0;

	if( 320 == width && 240 == height )
	{
		data[3] = 0;
	}
	else if( 640 == width && 480 == height )
	{
		data[3] = 1;
	}
	else
	{
		data[3] = 2;
		data[4] = ( width & 0xFF );
		data[5] = ( ( width >> 8 ) & 0xFF );
		data[6] = ( height & 0xFF );
		data[7] = ( ( height >> 8 ) & 0xFF );
	}

	return data;
}

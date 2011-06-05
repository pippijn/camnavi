#include "rec/robotino/com/v1/messages/IOControl.h"

using namespace rec::robotino::com::v1::messages;

QByteArray IOControl::encode( const rec::iocontrol::remotestate::SetState& setState )
{
	quint8* p;

	QByteArray data( 79, 0 );

	data[0] = 0; //IOControl message
	data[1] = 76; //76 data bytes
	data[2] = 0;

	data[3] = ( setState.isImageRequest ? 1 : 0 );
	data[4] = ( setState.shutdown ? 1 : 0 );
	data[5] = ( 0xFF & setState.imageServerPort );
	data[6] = ( setState.imageServerPort >> 8 );

	data[7] = setState.nstar_roomId;

	setState.toQDSAProtocol( reinterpret_cast<unsigned char*>( data.data() + 12 ) );

	data[59] = ( setState.setOdometry ? 1 : 0 );

	p = (quint8*)&setState.odometryX;
	data[60] = *(p++);
	data[61] = *(p++);
	data[62] = *(p++);
	data[63] = *(p++);

	p = (quint8*)&setState.odometryY;
	data[64] = *(p++);
	data[65] = *(p++);
	data[66] = *(p++);
	data[67] = *(p++);

	p = (quint8*)&setState.odometryPhi;
	data[68] = *(p++);
	data[69] = *(p++);
	data[70] = *(p++);
	data[71] = *(p++);

	p = (quint8*)&setState.nstar_ceilingCal;
	data[72] = *(p++);
	data[73] = *(p++);
	data[74] = *(p++);
	data[75] = *(p++);

	//data[76] = 0;
	//data[77] = 0;
	//data[78] = 0;

	return data;
}

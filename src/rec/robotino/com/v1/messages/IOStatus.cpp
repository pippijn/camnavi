#include "rec/robotino/com/v1/messages/IOStatus.h"

#include "rec/iocontrol/sercom/qdsa_protocol.h"

using namespace rec::robotino::com::v1::messages;

void IOStatus::decode( const QByteArray& data, rec::iocontrol::remotestate::SensorState* sensorState )
{
	qint16 tmp_int16;

	const unsigned char* datap = reinterpret_cast<const unsigned char*>( data.constData() );

	sensorState->serialLineUpdateFreqeuncy = *datap;
	sensorState->serialLineUpdateFreqeuncy |= ( *(datap+1) << 8 );
	
	sensorState->fromQDSAProtocol( datap+2 );

	//sequence number
	sensorState->nstar_sequenceNumber = *( datap+103 );
	sensorState->nstar_sequenceNumber |= ( *( datap+104 ) << 8 );
	sensorState->nstar_sequenceNumber |= ( *( datap+105 ) << 16 );
	sensorState->nstar_sequenceNumber |= ( *( datap+106 ) << 24 );

	sensorState->nstar_roomId = *( datap+107 );
	sensorState->nstar_numSpotsVisible = *( datap+108 );

	tmp_int16 = *( datap+109 );
	tmp_int16 |= ( *( datap+110 ) << 8 );

	sensorState->nstar_posX = tmp_int16;

	tmp_int16 = *( datap+111 );
	tmp_int16 |= ( *( datap+112 ) << 8 );

	sensorState->nstar_posY = tmp_int16;

	sensorState->nstar_posTheta = *( reinterpret_cast<const float*>( datap+113 ) );

	sensorState->nstar_magSpot0 = *( datap+117 );
	sensorState->nstar_magSpot0  |= ( *( datap+118 ) << 8 );

	sensorState->nstar_magSpot1 = *( datap+119 );
	sensorState->nstar_magSpot1  |= ( *( datap+120 ) << 8 );

	sensorState->isPassiveMode = ( *( datap+121 ) > 0 );

	sensorState->odometryX = *( reinterpret_cast<const float*>( datap+122 ) );
	sensorState->odometryY = *( reinterpret_cast<const float*>( datap+126 ) );
	sensorState->odometryPhi = *( reinterpret_cast<const float*>( datap+130 ) );

	//and 7 zeros
}

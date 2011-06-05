//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/iocontrol/sercom/qdsa_encoder.h"
#include "rec/iocontrol/sercom/qdsa_protocol.h"

#include <memory.h>
#include <assert.h>

#ifdef _MMGRDEBUG
#include "mmgr.h"
#include "mmgripp.h"
#endif

#ifndef WIN32
const unsigned int QDSA_Encoder::numMotors;
const unsigned int QDSA_Encoder::numDistances;
const unsigned int QDSA_Encoder::numBumpers;
const unsigned int QDSA_Encoder::numDO;
const unsigned int QDSA_Encoder::numDI;
const unsigned int QDSA_Encoder::numADC;
#endif //WIN32

std::map< std::string, PortIdentifier > QDSA_Encoder::_diMap;
std::map< std::string, PortIdentifier > QDSA_Encoder::_doMap;
std::map< std::string, ADIdentifier > QDSA_Encoder::_adiMap;

QDSA_Encoder::QDSA_Encoder( bool bitMode )
: SercomEncoder( NB_P2Q_FULL, NB_Q2P_FULL, bitMode )
{
	init();
}

QDSA_Encoder::QDSA_Encoder( unsigned char* p2q_buffer, unsigned int p2qSize, unsigned char* q2p_buffer, unsigned int q2pSize, bool bitMode )
: SercomEncoder( p2q_buffer, p2qSize, q2p_buffer, q2pSize, bitMode )
{
	init();
}

void QDSA_Encoder::init()
{
	_p2q_ws = _p2q_buffer + NB_START;
	_q2p_ws = _q2p_buffer + NB_START;

	if( _doMap.empty() )
	{
		_doMap[ "EMERGENCY_STOP" ] = PortIdentifier( M_DO, EMERGENCY_STOP, 1 );
		_doMap[ "POWER" ] = PortIdentifier( M_DO, POWER );
		_doMap[ "M_LED1" ] = PortIdentifier( M_DO, M_LED1 );

		_doMap[ "S0_BRAKE" ] = PortIdentifier( NUM_BYTES_P2M + S_DO, S_BRAKE );
		_doMap[ "S0_DO0" ] = PortIdentifier( NUM_BYTES_P2M + S_DO, S_DO0 );
		_doMap[ "S0_DO1" ] = PortIdentifier( NUM_BYTES_P2M + S_DO, S_DO1 );
		_doMap[ "S0_DO2" ] = PortIdentifier( NUM_BYTES_P2M + S_DO, S_DO2 );
		_doMap[ "S0_DO3" ] = PortIdentifier( NUM_BYTES_P2M + S_DO, S_DO3 );
		_doMap[ "S0_R0" ] = PortIdentifier( NUM_BYTES_P2M + S_DO, S_R0 );
		_doMap[ "S0_LED1" ] = PortIdentifier( NUM_BYTES_P2M + S_DO, S_LED1 );

		_doMap[ "S1_BRAKE" ] = PortIdentifier( NUM_BYTES_P2M + NUM_BYTES_P2S + S_DO, S_BRAKE );
		_doMap[ "S1_DO0" ] = PortIdentifier( NUM_BYTES_P2M + NUM_BYTES_P2S + S_DO, S_DO0 );
		_doMap[ "S1_DO1" ] = PortIdentifier( NUM_BYTES_P2M + NUM_BYTES_P2S + S_DO, S_DO1 );
		_doMap[ "S1_DO2" ] = PortIdentifier( NUM_BYTES_P2M + NUM_BYTES_P2S + S_DO, S_DO2 );
		_doMap[ "S1_DO3" ] = PortIdentifier( NUM_BYTES_P2M + NUM_BYTES_P2S + S_DO, S_DO3 );
		_doMap[ "S1_R0" ] = PortIdentifier( NUM_BYTES_P2M + NUM_BYTES_P2S + S_DO, S_R0 );
		_doMap[ "S1_LED1" ] = PortIdentifier( NUM_BYTES_P2M + S_DO, S_LED1 );

		_doMap[ "S2_BRAKE" ] = PortIdentifier( NUM_BYTES_P2M + 2*NUM_BYTES_P2S + S_DO, S_BRAKE );
		_doMap[ "S2_LED1" ] = PortIdentifier( NUM_BYTES_P2M + S_DO, S_LED1 );

		_doMap[ "S3_BRAKE" ] = PortIdentifier( NUM_BYTES_P2M + 3*NUM_BYTES_P2S + S_DO, S_BRAKE );
		_doMap[ "S3_LED1" ] = PortIdentifier( NUM_BYTES_P2M + S_DO, S_LED1 );

		_doMap[ "DO0" ] = _doMap[ "S0_DO0" ];
		_doMap[ "DO1" ] = _doMap[ "S0_DO1" ];
		_doMap[ "DO2" ] = _doMap[ "S0_DO2" ];
		_doMap[ "DO3" ] = _doMap[ "S0_DO3" ];
		_doMap[ "DO4" ] = _doMap[ "S1_DO0" ];
		_doMap[ "DO5" ] = _doMap[ "S1_DO1" ];
		_doMap[ "DO6" ] = _doMap[ "S1_DO2" ];
		_doMap[ "DO7" ] = _doMap[ "S1_DO3" ];
		_doMap[ "R0" ] = _doMap[ "S0_R0" ];
		_doMap[ "R1" ] = _doMap[ "S1_R0" ];
	}
	if( _diMap.empty() )
	{
		_diMap[ "S0_DI0" ] = PortIdentifier( NUM_BYTES_M2P + S_DI, S_DI0 );
		_diMap[ "S0_DI1" ] = PortIdentifier( NUM_BYTES_M2P + S_DI, S_DI1 );
		_diMap[ "S0_DI2" ] = PortIdentifier( NUM_BYTES_M2P + S_DI, S_DI2 );
		_diMap[ "S0_DI3" ] = PortIdentifier( NUM_BYTES_M2P + S_DI, S_DI3 );
		_diMap[ "S0_BUMPER" ] = PortIdentifier( NUM_BYTES_M2P + S_DI, S_BUMPER );

		_diMap[ "S1_DI0" ] = PortIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_DI, S_DI0 );
		_diMap[ "S1_DI1" ] = PortIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_DI, S_DI1 );
		_diMap[ "S1_DI2" ] = PortIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_DI, S_DI2 );
		_diMap[ "S1_DI3" ] = PortIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_DI, S_DI3 );

		_diMap[ "S2_DI0" ] = PortIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_DI, S_DI0 );
		_diMap[ "S2_DI1" ] = PortIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_DI, S_DI1 );
		_diMap[ "S2_DI2" ] = PortIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_DI, S_DI2 );
		_diMap[ "S2_DI3" ] = PortIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_DI, S_DI3 );

		_diMap[ "S3_DI0" ] = PortIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_DI, S_DI0 );
		_diMap[ "S3_DI1" ] = PortIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_DI, S_DI1 );
		_diMap[ "S3_DI2" ] = PortIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_DI, S_DI2 );
		_diMap[ "S3_DI3" ] = PortIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_DI, S_DI3 );

		_diMap[ "Bumper" ] = _diMap[ "S0_BUMPER" ];

		_diMap[ "DI0" ] = _diMap[ "S0_DI0" ];
		_diMap[ "DI1" ] = _diMap[ "S0_DI1" ];
		_diMap[ "DI2" ] = _diMap[ "S0_DI2" ];
		_diMap[ "DI3" ] = _diMap[ "S0_DI3" ];
		_diMap[ "DI4" ] = _diMap[ "S1_DI0" ];
		_diMap[ "DI5" ] = _diMap[ "S1_DI1" ];
		_diMap[ "DI6" ] = _diMap[ "S1_DI2" ];
		_diMap[ "DI7" ] = _diMap[ "S1_DI3" ];
	}
	if( _adiMap.empty() )
	{
		_adiMap[ "M_AD0" ] = ADIdentifier( M_AD0_H, M_AD_L0, 0 );
		_adiMap[ "M_AD1" ] = ADIdentifier( M_AD1_H, M_AD_L0, 1 );
		_adiMap[ "M_AD2" ] = ADIdentifier( M_AD2_H, M_AD_L0, 2 );
		_adiMap[ "M_AD3" ] = ADIdentifier( M_AD3_H, M_AD_L0, 3 );
		_adiMap[ "M_AD4" ] = ADIdentifier( M_AD4_H, M_AD_L1, 0 );
		_adiMap[ "M_AD5" ] = ADIdentifier( M_AD5_H, M_AD_L1, 1 );
		_adiMap[ "M_AD6" ] = ADIdentifier( M_AD6_H, M_AD_L1, 2 );
		_adiMap[ "M_AD7" ] = ADIdentifier( M_AD7_H, M_AD_L1, 3 );

		_adiMap[ "S0_AD0" ] = ADIdentifier( NUM_BYTES_M2P + S_AD0_H, NUM_BYTES_M2P + S_AD_L0, 0 );
		_adiMap[ "S0_AD1" ] = ADIdentifier( NUM_BYTES_M2P + S_AD1_H, NUM_BYTES_M2P + S_AD_L0, 1 );
		_adiMap[ "S0_AD2" ] = ADIdentifier( NUM_BYTES_M2P + S_AD2_H, NUM_BYTES_M2P + S_AD_L0, 2 );
		_adiMap[ "S0_AD3" ] = ADIdentifier( NUM_BYTES_M2P + S_AD3_H, NUM_BYTES_M2P + S_AD_L0, 3 );
		_adiMap[ "S0_AD4" ] = ADIdentifier( NUM_BYTES_M2P + S_AD4_H, NUM_BYTES_M2P + S_AD_L1, 0 );
		_adiMap[ "S0_AD5" ] = ADIdentifier( NUM_BYTES_M2P + S_AD5_H, NUM_BYTES_M2P + S_AD_L1, 1 );
		_adiMap[ "S0_AD6" ] = ADIdentifier( NUM_BYTES_M2P + S_AD6_H, NUM_BYTES_M2P + S_AD_L1, 2 );
		_adiMap[ "S0_AD7" ] = ADIdentifier( NUM_BYTES_M2P + S_AD7_H, NUM_BYTES_M2P + S_AD_L1, 3 );

		_adiMap[ "S1_AD0" ] = ADIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD0_H, NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD_L0, 0 );
		_adiMap[ "S1_AD1" ] = ADIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD1_H, NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD_L0, 1 );
		_adiMap[ "S1_AD2" ] = ADIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD2_H, NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD_L0, 2 );
		_adiMap[ "S1_AD3" ] = ADIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD3_H, NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD_L0, 3 );
		_adiMap[ "S1_AD4" ] = ADIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD4_H, NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD_L1, 0 );
		_adiMap[ "S1_AD5" ] = ADIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD5_H, NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD_L1, 1 );
		_adiMap[ "S1_AD6" ] = ADIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD6_H, NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD_L1, 2 );
		_adiMap[ "S1_AD7" ] = ADIdentifier( NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD7_H, NUM_BYTES_M2P + NUM_BYTES_S2P + S_AD_L1, 3 );

		_adiMap[ "S2_AD0" ] = ADIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD0_H, NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD_L0, 0 );
		_adiMap[ "S2_AD1" ] = ADIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD1_H, NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD_L0, 1 );
		_adiMap[ "S2_AD2" ] = ADIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD2_H, NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD_L0, 2 );
		_adiMap[ "S2_AD3" ] = ADIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD3_H, NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD_L0, 3 );
		_adiMap[ "S2_AD4" ] = ADIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD4_H, NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD_L1, 0 );
		_adiMap[ "S2_AD5" ] = ADIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD5_H, NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD_L1, 1 );
		_adiMap[ "S2_AD6" ] = ADIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD6_H, NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD_L1, 2 );
		_adiMap[ "S2_AD7" ] = ADIdentifier( NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD7_H, NUM_BYTES_M2P + 2*NUM_BYTES_S2P + S_AD_L1, 3 );

		_adiMap[ "S3_AD0" ] = ADIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD0_H, NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD_L0, 0 );
		_adiMap[ "S3_AD1" ] = ADIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD1_H, NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD_L0, 1 );
		_adiMap[ "S3_AD2" ] = ADIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD2_H, NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD_L0, 2 );
		_adiMap[ "S3_AD3" ] = ADIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD3_H, NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD_L0, 3 );
		_adiMap[ "S3_AD4" ] = ADIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD4_H, NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD_L1, 0 );
		_adiMap[ "S3_AD5" ] = ADIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD5_H, NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD_L1, 1 );
		_adiMap[ "S3_AD6" ] = ADIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD6_H, NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD_L1, 2 );
		_adiMap[ "S3_AD7" ] = ADIdentifier( NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD7_H, NUM_BYTES_M2P + 3*NUM_BYTES_S2P + S_AD_L1, 3 );

		_adiMap[ "IR1" ] = _adiMap[ "S2_AD2" ];
		_adiMap[ "IR2" ] = _adiMap[ "S2_AD1" ];
		_adiMap[ "IR3" ] = _adiMap[ "S1_AD7" ];
		_adiMap[ "IR4" ] = _adiMap[ "S1_AD2" ];
		_adiMap[ "IR5" ] = _adiMap[ "S1_AD1" ];
		_adiMap[ "IR6" ] = _adiMap[ "S0_AD1" ];
		_adiMap[ "IR7" ] = _adiMap[ "S0_AD2" ];
		_adiMap[ "IR8" ] = _adiMap[ "S0_AD7" ];
		_adiMap[ "IR9" ] = _adiMap[ "S2_AD7" ];

		_adiMap[ "AIN0" ] = _adiMap[ "S0_AD3" ];
		_adiMap[ "AIN1" ] = _adiMap[ "S0_AD4" ];
		_adiMap[ "AIN2" ] = _adiMap[ "S0_AD5" ];
		_adiMap[ "AIN3" ] = _adiMap[ "S0_AD6" ];
		_adiMap[ "AIN4" ] = _adiMap[ "S1_AD3" ];
		_adiMap[ "AIN5" ] = _adiMap[ "S1_AD4" ];
		_adiMap[ "AIN6" ] = _adiMap[ "S1_AD5" ];
		_adiMap[ "AIN7" ] = _adiMap[ "S1_AD6" ];
	}

	if( isBufferOwner() )
	{
		reset();
	}
}

QDSA_Encoder* QDSA_Encoder::deepCopy() const
{
	QDSA_Encoder* e = new QDSA_Encoder( this->_bitMode );
	e->clone( this );
	return e;
}

void QDSA_Encoder::reset()
{
	lock lk1( _p2q_mutex );
	lock lk2( _q2p_mutex );

	memset( (void*)_p2q_buffer, 0, p2qSize() );
	memset( (void*)_q2p_buffer, 0, q2pSize() );

	// initialize start sequence
	_p2q_buffer[ 0 ] = START0;
	_p2q_buffer[ 1 ] = START1;
	_p2q_buffer[ 2 ] = START2;

	_p2q_buffer[ p2qSize() - 3 ] = STOP0;
	_p2q_buffer[ p2qSize() - 2 ] = STOP1;
	_p2q_buffer[ p2qSize() - 1 ] = STOP2;

	std::map< std::string, PortIdentifier >::iterator iter = _doMap.begin();
	while( iter != _doMap.end() )
	{
		PortIdentifier& pi = iter->second;
		if( pi.initial() > 0 )
		{
			_p2q_ws[ pi.byte() ] |= _BV( pi.bit() );
		}
		iter++;
	}

	for( unsigned int i=0; i<numMotors; i++ )
	{
		set_KP_i( i, KP_DEFAULT );
		set_KI_i( i, KI_DEFAULT );
		set_KD_i( i, KD_DEFAULT );
		set_DV_i( i, 0 );
	}
}

unsigned int QDSA_Encoder::get_FirmwareVersion() const
{
	boost::uint8_t value = 0;

	lock lk( _q2p_mutex );
	if( bit_is_set( _q2p_ws[ NUM_BYTES_M2P + S_MISCIN ], FIRMWAREVERSION0 ) )
	{
		value += 1;
	}
	if( bit_is_set( _q2p_ws[ NUM_BYTES_M2P + S_MISCIN ], FIRMWAREVERSION1 ) )
	{
		value += 2;
	}
	if( bit_is_set( _q2p_ws[ NUM_BYTES_M2P + S_MISCIN ], FIRMWAREVERSION2 ) )
	{
		value += 4;
	}
	if( bit_is_set( _q2p_ws[ NUM_BYTES_M2P + S_MISCIN ], FIRMWAREVERSION3 ) )
	{
		value += 8;
	}

	switch( value )
	{
	case 0:
		return FIRMWAREVERSIONMAPPGING0;
	case 1:
		return FIRMWAREVERSIONMAPPGING1;
	case 2:
		return FIRMWAREVERSIONMAPPGING2;
	case 3:
		return FIRMWAREVERSIONMAPPGING3;
	case 4:
		return FIRMWAREVERSIONMAPPGING4;
	case 5:
		return FIRMWAREVERSIONMAPPGING5;
	case 6:
		return FIRMWAREVERSIONMAPPGING6;
	case 7:
		return FIRMWAREVERSIONMAPPGING7;
	case 8:
		return FIRMWAREVERSIONMAPPGING8;
	case 9:
		return FIRMWAREVERSIONMAPPGING9;
	case 10:
		return FIRMWAREVERSIONMAPPGING10;
	case 11:
		return FIRMWAREVERSIONMAPPGING11;
	case 12:
		return FIRMWAREVERSIONMAPPGING12;
	case 13:
		return FIRMWAREVERSIONMAPPGING13;
	case 14:
		return FIRMWAREVERSIONMAPPGING14;
	case 15:
		return FIRMWAREVERSIONMAPPGING15;

	default:
		return 0;
	}
}

void QDSA_Encoder::set_Brake( boost::uint8_t motor, bool on )
{
	lock lk( _p2q_mutex );
	setBit( &_p2q_ws[NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_DO], S_BRAKE, ( on ? 0 : 1 ) );
}

bool QDSA_Encoder::get_Brake( boost::uint8_t motor ) const
{
	return !bit_is_set( _p2q_ws[NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_DO], S_BRAKE );
}

void QDSA_Encoder::set_DO( const std::string& name, boost::uint8_t value )
{
	if( _doMap.find( name ) != _doMap.end() )
	{
		lock lk( _p2q_mutex );
		PortIdentifier pi = _doMap[ name ];
		setBit( &_p2q_ws[ pi.byte() ], pi.bit(), value );
	}
}

void QDSA_Encoder::set_DI( const std::string& name, boost::uint8_t value )
{
	if( _diMap.find( name ) != _diMap.end() )
	{
		lock lk( _q2p_mutex );
		PortIdentifier pi = _diMap[ name ];
		setBit( &_q2p_ws[ pi.byte() ], pi.bit(), value );
	}
}

unsigned char QDSA_Encoder::ackChar() const
{
	return ACK_CHAR;
}

unsigned char QDSA_Encoder::restartChar() const
{
	return RESTART_CHAR;
}

unsigned int QDSA_Encoder::bytesPerPacketQ2P() const
{
	// specific for 16550A
	return NUM_BYTES_PER_PACKET;
}

unsigned int QDSA_Encoder::bytesPerPacketP2Q() const
{
	return 0;
}

bool QDSA_Encoder::checkQ2P( const unsigned char* buffer, unsigned int relevantByte )
{
	static const unsigned char start_sequence[NB_START] = {START0, START1, START2};
	static const unsigned char stop_sequence[NB_STOP] = {STOP0, STOP1, STOP2};

	if( relevantByte < NB_START )
	{
		if( buffer[relevantByte] != start_sequence[relevantByte] )
		{
			return false;
		}
	}
	if( relevantByte >= NB_START + NUM_BYTES_Q2P )
	{
		if( buffer[relevantByte] != stop_sequence[relevantByte - NB_START - NUM_BYTES_Q2P] )
		{
			return false;
		}
	}
	return true;
}

boost::int16_t QDSA_Encoder::get_DV( boost::uint8_t motor ) const
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return 0;
	}

	lock lk( _p2q_mutex );

	boost::int16_t value;
	value = _p2q_ws[NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_DV];

	if( bit_is_set( _p2q_ws[NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_MISCOUT], DV_DIR ) == false )
	{
		value = -value;
	}

	return value;
}

void QDSA_Encoder::set_DV( boost::uint8_t motor, boost::int16_t value )
{
	lock lk( _p2q_mutex );
	set_DV_i( motor, value );
}

void QDSA_Encoder::set_DV_i( boost::uint8_t motor, boost::int16_t value )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}


	unsigned int dir;
	unsigned char vabs;

	if( value > 255 )
	{
		value = 255;
	}
	if( value < -255 )
	{
		value = -255;
	}

	if( value >= 0 )
	{
		vabs = static_cast< unsigned char >( value );
		dir = 1;
	}
	else
	{
		vabs = -value;
		dir = 0;
	}

	setBit( &_p2q_ws[NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_MISCOUT], DV_DIR, dir );
	_p2q_ws[NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_DV] = vabs;
}

void QDSA_Encoder::set_DP( boost::uint8_t motor, boost::int32_t value )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}

	lock lk( _p2q_mutex );

	boost::uint8_t* p = (boost::uint8_t*)&value;

	_p2q_ws[NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_DP_0] = *p++;
	_p2q_ws[NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_DP_1] = *p++;
	_p2q_ws[NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_DP_2] = *p++;
	_p2q_ws[NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_DP_3] = *p++;
}

void QDSA_Encoder::set_MODE( boost::uint8_t motor, boost::uint8_t value )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}
	lock lk( _p2q_mutex );
	setBit( &_p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_MISCOUT ], MODE, value );
}

void QDSA_Encoder::set_KP( boost::uint8_t motor, boost::uint8_t value )
{
	lock lk( _p2q_mutex );
	set_KP_i( motor, value );
}

void QDSA_Encoder::set_KP_i( boost::uint8_t motor, boost::uint8_t value )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}
	_p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_KP ] = value;
}

boost::uint8_t QDSA_Encoder::get_KP( boost::uint8_t motor ) const
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return 0;
	}

	lock lk( _p2q_mutex );
	return _p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_KP ];
}

void QDSA_Encoder::set_KI( boost::uint8_t motor, boost::uint8_t value )
{
	lock lk( _p2q_mutex );
	set_KI_i( motor, value );
}

void QDSA_Encoder::set_KI_i( boost::uint8_t motor, boost::uint8_t value )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}
	_p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_KI ] = value;
}

boost::uint8_t QDSA_Encoder::get_KI( boost::uint8_t motor ) const
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return 0;
	}

	lock lk( _p2q_mutex );
	return _p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_KI ];
}

void QDSA_Encoder::set_KD( boost::uint8_t motor, boost::uint8_t value )
{
	lock lk( _p2q_mutex );
	set_KD_i( motor, value );
}

void QDSA_Encoder::set_KD_i( boost::uint8_t motor, boost::uint8_t value )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}
	_p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_KD ] = value;
}

boost::uint8_t QDSA_Encoder::get_KD( boost::uint8_t motor ) const
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return 0;
	}

	lock lk( _p2q_mutex );
	return _p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_KD ];
}

void QDSA_Encoder::set_Shutdown( bool value )
{
	if( value )
	{
		*(_p2q_ws) = 0xff;
	}
	else
	{
		*(_p2q_ws) = 0;
	}
	//setBit( &_p2q_ws[ M_DO ], POWER, value ? 0 : 1 );
}

boost::uint8_t QDSA_Encoder::get_DI( const std::string& name ) const
{
	if( _diMap.find( name ) != _diMap.end() )
	{
		lock lk( _q2p_mutex );
		PortIdentifier pi = _diMap[ name ];
		return bit_is_set( _q2p_ws[ pi.byte() ], pi.bit() );
	}
	else
	{
		return 0;
	}
}

bool QDSA_Encoder::get_DO( const std::string& name ) const
{
	if( _doMap.find( name ) != _doMap.end() )
	{
		lock lk( _q2p_mutex );
		PortIdentifier pi = _doMap[ name ];
		return bit_is_set( _p2q_ws[ pi.byte() ], pi.bit() );
	}
	else
	{
		return false;
	}
}

void QDSA_Encoder::set_AV( boost::uint8_t motor, boost::int16_t value )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}

	lock lk( _q2p_mutex );

	unsigned int dir;
	unsigned char vabs;

	if( value > 255 )
	{
		value = 255;
	}
	if( value < -255 )
	{
		value = -255;
	}

	if( value >= 0 )
	{
		vabs = static_cast< unsigned char >( value );
		dir = 1;
	}
	else
	{
		vabs = -value;
		dir = 0;
	}

	setBit( &_q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_MISCIN], AV_DIR, dir );
	_q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_AV] = vabs;
}

boost::int16_t QDSA_Encoder::get_AV( boost::uint8_t motor ) const
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return 0;
	}

	lock lk( _q2p_mutex );

	boost::int16_t value = 0;

	value = _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_AV];
	if( !bit_is_set( _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_MISCIN], AV_DIR ) )
	{
		value = -value;
	}

	return value; 

}

void QDSA_Encoder::set_AP( boost::uint8_t motor, boost::int32_t pos )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}
	lock lk( _q2p_mutex );

	boost::uint8_t* p = (boost::uint8_t*)&pos;

	_q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_AP_0] = *p++;
	_q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_AP_1] = *p++;
	_q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_AP_2] = *p++;
	_q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_AP_3] = *p;
}

boost::int32_t QDSA_Encoder::get_AP( boost::uint8_t motor ) const
{ 
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return 0;
	}
	lock lk( _q2p_mutex );

	boost::int32_t pos;
	boost::uint8_t* p = (boost::uint8_t*)&pos;

	*p++ = _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_AP_0];
	*p++ = _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_AP_1];
	*p++ = _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_AP_2];
	*p   = _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_AP_3];

	return pos;
}

void QDSA_Encoder::set_AD( const std::string& name, boost::uint16_t value )
{
	if( _adiMap.find( name ) != _adiMap.end() )
	{
		if( value > 0x3FF )
		{
			value = 0x3FF;
		}
		lock lk( _q2p_mutex );
		ADIdentifier adi = _adiMap[ name ];

		_q2p_ws[ adi.hibyte() ] = ( value >> 2 );
		_q2p_ws[ adi.lobyte() ] |= ( (value & 0x3) << ( 2*adi.channel() ) );
	}
}

boost::uint16_t QDSA_Encoder::get_AD( const std::string& name ) const
{
	if( _adiMap.find( name ) != _adiMap.end() )
	{
		lock lk( _q2p_mutex );
		ADIdentifier adi = _adiMap[ name ];
		boost::uint16_t value;
		value = ( _q2p_ws[ adi.hibyte() ] << 2 );
		value |= ( 0x3 & ( _q2p_ws[ adi.lobyte() ] >> ( 2*adi.channel() ) ) );
		return value;
	}
	else
	{
		return 0;
	}
}

std::list< std::string > QDSA_Encoder::doMapKeys() const
{
	std::list< std::string > l;
	std::map< std::string, PortIdentifier >::const_iterator ci = _doMap.begin();
	while( ci != _doMap.end() )
	{
		l.push_back( ci->first );
		ci++;
	}
	return l;
}

std::list< std::string > QDSA_Encoder::diMapKeys() const
{
	std::list< std::string > l;
	std::map< std::string, PortIdentifier >::const_iterator ci = _diMap.begin();
	while( ci != _diMap.end() )
	{
		l.push_back( ci->first );
		ci++;
	}
	return l;
}

std::list< std::string > QDSA_Encoder::adiMapKeys() const
{
	std::list< std::string > l;
	std::map< std::string, ADIdentifier >::const_iterator ci = _adiMap.begin();
	while( ci != _adiMap.end() )
	{
		l.push_back( ci->first );
		ci++;
	}
	return l;
}

int QDSA_Encoder::get_Error( boost::uint8_t motor ) const
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return NoError;
	}
	lock lk( _q2p_mutex );

	if( bit_is_set( _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_MISCIN], SPIERROR0 ) )
	{
		return SPI0Error;
	}
	else if( bit_is_set( _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_MISCIN], SPIERROR1 ) )
	{
		return SPI1Error;
	}
	else
	{
		return NoError;
	}
}

boost::uint8_t QDSA_Encoder::get_MasterTime() const
{
	lock lk( _q2p_mutex );
	return _q2p_ws[M_TIME];
}

void QDSA_Encoder::set_MotorTime( boost::uint8_t motor, boost::uint32_t time )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}
	lock lk( _q2p_mutex );

	boost::uint8_t* p = (boost::uint8_t*)&time;

	_q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_TIME_0] = *p++;
	_q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_TIME_1] = *p++;
	_q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_TIME_2] = *p++;
	_q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_TIME_3] = *p;
}

boost::uint32_t QDSA_Encoder::get_MotorTime( boost::uint8_t motor ) const
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return 0;
	}
	lock lk( _q2p_mutex );

	boost::uint32_t time = 0;
	boost::uint8_t* p = (boost::uint8_t*)&time;

	*p++ = _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_TIME_0];
	*p++ = _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_TIME_1];
	*p++ = _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_TIME_2];
	*p   = _q2p_ws[NUM_BYTES_M2P + (NUM_BYTES_S2P*motor) + S_TIME_3];

	return time;
}

void QDSA_Encoder::set_ResetPosition( unsigned int motor, bool reset )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}
	lock lk( _p2q_mutex );
	setBit( &_p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_MISCOUT ], RESET_POS, reset );
}

bool QDSA_Encoder::get_ResetPosition( unsigned int motor ) const
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return false;
	}
	return bit_is_set( _p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_MISCOUT ], RESET_POS );
}

void QDSA_Encoder::set_ResetMotorTime( unsigned int motor, bool reset )
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return;
	}
	lock lk( _p2q_mutex );
	setBit( &_p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_MISCOUT ], RESET_TIME, reset );
}

bool QDSA_Encoder::get_ResetMotorTime( unsigned int motor ) const
{
	//die Motoren werden mit 0 beginnend gezählt
	if( motor >= numMotors )
	{
		return false;
	}
	return bit_is_set( _p2q_ws[ NUM_BYTES_P2M + (NUM_BYTES_P2S*motor) + S_MISCOUT ], RESET_TIME );
}



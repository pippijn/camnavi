//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_IOCONTROL_REMOTESTATE_SENSORSTATE_H_
#define _REC_IOCONTROL_REMOTESTATE_SENSORSTATE_H_

#include <cstring>

namespace rec
{
	namespace iocontrol
	{
		namespace remotestate
		{
			/**
			@brief The SensorState is the collection of all sensor readings received from Robotino.
			*/
			class SensorState
			{
			public:
				SensorState();

				SensorState( const SensorState& other );

				SensorState& operator=( const SensorState& other );

				void toQDSAProtocol( unsigned char* ) const;

				bool fromQDSAProtocol( const unsigned char* );

				/**
				Reset this state to default values.
				*/
				void reset();

				float powerOutputCurrent;
				unsigned short powerOutputRawCurrent;

				int encoderInputPosition;
				int encoderInputVelocity;

				//Velocity/Position
				float actualVelocity[3];
				int actualPosition[3];
				float motorCurrent[3];
				unsigned short rawMotorCurrent[3];

				//digital inputs
				bool dIn[8];
				//analog inputs
				float aIn[8];
				//distance sensors
				float distanceSensor[9];
				//bumper
				bool bumper;

				float current;
				float voltage;

				bool isPassiveMode;

				bool isGripperOpened;
				bool isGripperClosed;

				float odometryX;
				float odometryY;
				float odometryPhi;

				unsigned int serialLineUpdateFreqeuncy;

				unsigned int firmwareVersion;

				unsigned int sequenceNumber;

				//NorthStar **********
				unsigned int nstar_sequenceNumber;
				unsigned int nstar_roomId;
				unsigned int nstar_numSpotsVisible;
				float nstar_posX;
				float nstar_posY;
				float nstar_posTheta;
				unsigned int nstar_magSpot0;
				unsigned int nstar_magSpot1;
				//NorthStar **********

				//EA09 firmware version
				//If this is {0,0,0} we have a EA05
				unsigned char firmware_version[2];

			private:
				void copy( const SensorState& other );
			};

			inline SensorState::SensorState()
			{
				reset();
			}

			inline SensorState::SensorState( const SensorState& other )
			{
				copy( other );
			}

			inline SensorState& SensorState::operator=( const SensorState& other )
			{
				copy( other );
				return *this;
			}

			inline void SensorState::toQDSAProtocol( unsigned char* buffer ) const
			{
				unsigned short distance[9];
				unsigned short ad[8];
				short avint[3];
				unsigned short currentint;
				unsigned short voltageint;
				unsigned char* uint8p;
				unsigned int* uint32p;
				int tmpint;

				for( unsigned int i=0; i<9; ++i )
				{
					distance[i] = static_cast<unsigned short>( 1024.0f / 2.55f * distanceSensor[i] );
				}

				for( unsigned int i=0; i<8; ++i )
				{
					ad[i] = static_cast<unsigned short>( 1024.0f / 10.0f * aIn[i] );
				}

				for( unsigned int i=0; i<3; ++i )
				{
					avint[i] = static_cast<short>( actualVelocity[i] / 27.0f ); // 900 * 60 / 2000 = 27
					if( avint[i] > 255 )
					{
						avint[i] = 255;
					}
					else if( avint[i] < -255 )
					{
						avint[i] = -255;
					}
				}

				currentint = static_cast<unsigned short>( current * 61.44f ); 
				voltageint = static_cast<unsigned short>( voltage * 36.10f );

				*buffer       = 'R';
				*(buffer+1)   = 'E';
				*(buffer+2)   = 'C';
				*(buffer+98)  = 'r';
				*(buffer+99)  = 'e';
				*(buffer+100) = 'c';

				//master start at byte 3
				*(buffer+3) = ( ( currentint >> 2 ) & 0xFF );
				*(buffer+5) = ( ( voltageint >> 2 ) & 0xFF );

				*(buffer+11) =  ( currentint & 0x3 );
				*(buffer+11) |=  ( ( voltageint & 0x3 ) << 4 );

				//slave 0
				*(buffer+14) = ( ( rawMotorCurrent[0] >> 2 ) & 0xFF );
				*(buffer+15) = ( ( distance[5] >> 2 ) & 0xFF );
				*(buffer+16) = ( ( distance[6] >> 2 ) & 0xFF );
				*(buffer+17) = ( ( ad[0] >> 2 ) & 0xFF );
				*(buffer+18) = ( ( ad[1] >> 2 ) & 0xFF );
				*(buffer+19) = ( ( ad[2] >> 2 ) & 0xFF );
				*(buffer+20) = ( ( ad[3] >> 2 ) & 0xFF );
				*(buffer+21) = ( ( distance[7] >> 2 ) & 0xFF );

				*(buffer+22) =  ( rawMotorCurrent[0] & 0x3 );
				*(buffer+22) |=  ( ( distance[5] & 0x3 ) << 2 );
				*(buffer+22) |= ( ( distance[6] & 0x3 ) << 4 );
				*(buffer+22) |= ( ( ad[0] & 0x3 ) << 6 );

				*(buffer+23) =  ( ad[1] & 0x3 );
				*(buffer+23) |=  ( ( ad[2] & 0x3 ) << 2 );
				*(buffer+23) |= ( ( ad[3] & 0x3 ) << 4 );
				*(buffer+23) |= ( ( distance[7] & 0x3 ) << 6 );

				*(buffer+24) = ( ( ( firmwareVersion & 1<<3 ) > 0 ) ? 1<<4 : 0 );
				*(buffer+24) = ( ( ( firmwareVersion & 1<<2 ) > 0 ) ? 1<<5 : 0 );
				*(buffer+24) = ( ( ( firmwareVersion & 1<<1 ) > 0 ) ? 1<<6 : 0 );
				*(buffer+24) = ( ( ( firmwareVersion & 1 ) > 0 ) ? 1<<7 : 0 );

				if( avint[0] >= 0 )
				{
					*(buffer+24) = 0x1;
					*(buffer+25) = static_cast<unsigned char>( avint[0] );
				}
				else
				{
					*(buffer+25) = static_cast<unsigned char>( -avint[0] );
				}

				tmpint = -actualPosition[0];
				uint8p = reinterpret_cast<unsigned char*>( &tmpint );
				*(buffer+26) = *(uint8p++);
				*(buffer+27) = *(uint8p++);
				*(buffer+28) = *(uint8p++);
				*(buffer+29) = *(uint8p);

				*(buffer+30) = ( dIn[0] ? 1 : 0 );
				*(buffer+30) |= ( dIn[1] ? 1<<1 : 0 );
				*(buffer+30) |= ( dIn[2] ? 1<<2 : 0 );
				*(buffer+30) |= ( dIn[3] ? 1<<3 : 0 );
				*(buffer+30) |= ( bumper ? 1<<4 : 0 );

				uint32p = reinterpret_cast<unsigned int*>( buffer+31 );
				*uint32p = sequenceNumber;

				//slave 1
				*(buffer+35) = ( ( rawMotorCurrent[1] >> 2 ) & 0xFF );
				*(buffer+36) = ( ( distance[4] >> 2 ) & 0xFF );
				*(buffer+37) = ( ( distance[3] >> 2 ) & 0xFF );
				*(buffer+38) = ( ( ad[4] >> 2 ) & 0xFF );
				*(buffer+39) = ( ( ad[5] >> 2 ) & 0xFF );
				*(buffer+40) = ( ( ad[6] >> 2 ) & 0xFF );
				*(buffer+41) = ( ( ad[7] >> 2 ) & 0xFF );
				*(buffer+42) = ( ( distance[2] >> 2 ) & 0xFF );

				*(buffer+43) =  ( rawMotorCurrent[1] & 0x3 );
				*(buffer+43) |=  ( ( distance[4] & 0x3 ) << 2 );
				*(buffer+43) |= ( ( distance[3] & 0x3 ) << 4 );
				*(buffer+43) |= ( ( ad[4] & 0x3 ) << 6 );

				*(buffer+44) =  ( ad[5] & 0x3 );
				*(buffer+44) |=  ( ( ad[6] & 0x3 ) << 2 );
				*(buffer+44) |= ( ( ad[7] & 0x3 ) << 4 );
				*(buffer+44) |= ( ( distance[2] & 0x3 ) << 6 );

				if( avint[1] >= 0 )
				{
					*(buffer+45) = 0x1;
					*(buffer+46) = static_cast<unsigned char>( avint[1] );
				}
				else
				{
					*(buffer+46) = static_cast<unsigned char>( -avint[1] );
				}

				tmpint = -actualPosition[1];
				uint8p = reinterpret_cast<unsigned char*>( &tmpint );
				*(buffer+47) = *(uint8p++);
				*(buffer+48) = *(uint8p++);
				*(buffer+49) = *(uint8p++);
				*(buffer+50) = *(uint8p);

				*(buffer+51) = ( dIn[4] ? 1 : 0 );
				*(buffer+51) |= ( dIn[5] ? 1<<1 : 0 );
				*(buffer+51) |= ( dIn[6] ? 1<<2 : 0 );
				*(buffer+51) |= ( dIn[7] ? 1<<3 : 0 );

				//slave 2
				*(buffer+56) = ( ( rawMotorCurrent[2] >> 2 ) & 0xFF );
				*(buffer+57) = ( ( distance[1] >> 2 ) & 0xFF );
				*(buffer+58) = ( ( distance[0] >> 2 ) & 0xFF );
				*(buffer+63) = ( ( distance[8] >> 2 ) & 0xFF );

				*(buffer+64) =  ( rawMotorCurrent[2] & 0x3 );
				*(buffer+64) |=  ( ( distance[1] & 0x3 ) << 2 );
				*(buffer+64) |= ( ( distance[0] & 0x3 ) << 4 );

				*(buffer+65) =  ( ( distance[8] & 0x3 ) << 6 );

				if( avint[2] >= 0 )
				{
					*(buffer+66) = 0x1;
					*(buffer+67) = static_cast<unsigned char>( avint[2] );
				}
				else
				{
					*(buffer+67) = static_cast<unsigned char>( -avint[2] );
				}

				tmpint = -actualPosition[2];
				uint8p = reinterpret_cast<unsigned char*>( &tmpint );
				*(buffer+68) = *(uint8p++);
				*(buffer+69) = *(uint8p++);
				*(buffer+70) = *(uint8p++);
				*(buffer+71) = *(uint8p);

				//slave 3
				*(buffer+77) = ( ( powerOutputRawCurrent >> 2 ) & 0xFF );
				*(buffer+85) = ( powerOutputRawCurrent & 0x3 );

				if( encoderInputVelocity >= 0 )
				{
					*(buffer+87) = 0x1;
					*(buffer+88) = encoderInputVelocity;
				}
				else
				{
					*(buffer+88) = -encoderInputVelocity;
				}

				tmpint = -encoderInputPosition;
				uint8p = reinterpret_cast<unsigned char*>( &tmpint );
				*(buffer+89) = *(uint8p++);
				*(buffer+90) = *(uint8p++);
				*(buffer+91) = *(uint8p++);
				*(buffer+92) = *(uint8p);
			}

			inline bool SensorState::fromQDSAProtocol( const unsigned char* data )
			{
				unsigned short distance[9] = {0,0,0,0,0,0,0,0,0};
				unsigned short ad[8] = {0,0,0,0,0,0,0,0};
				short avint[3] = {0,0,0};
				unsigned short currentint;
				unsigned short voltageint;
				unsigned char* uint8p;

				if( *(data)     != 'R' ||
					*(data+1)   != 'E' ||
					*(data+2)   != 'C' ||
					*(data+98)  != 'r' ||
					*(data+99)  != 'e' ||
					*(data+100) != 'c' )
				{
					return false;
				}

				//master start at byte 3
				currentint = ( *(data+3) << 2);
				voltageint = ( *(data+5) << 2);

				firmware_version[0] = *(data+7);
				firmware_version[1] = *(data+8);
				firmware_version[2] = *(data+9);

				currentint |= ( *(data+11) & 0x3 );
				voltageint |= ( ( *(data+11) >> 4 ) & 0x3 );

				//slave 0 start at byte 14
				rawMotorCurrent[0] = ( *(data+14) << 2);
				distance[5] = ( *(data+15) << 2);
				distance[6] = ( *(data+16) << 2);
				ad[0] = ( *(data+17) << 2);
				ad[1] = ( *(data+18) << 2);
				ad[2] = ( *(data+19) << 2);
				ad[3] = ( *(data+20) << 2);
				distance[7] = ( *(data+21) << 2);

				rawMotorCurrent[0] |= ( *(data+22) & 0x3 );
				distance[5] |= ( ( *(data+22) >> 2 ) & 0x3 );
				distance[6] |= ( ( *(data+22) >> 4 ) & 0x3 );
				ad[0] |= ( ( *(data+22) >> 6 ) & 0x3 );

				ad[1] |= ( ( *(data+23) ) & 0x3 );
				ad[2] |= ( ( *(data+23) >> 2 ) & 0x3 );
				ad[3] |= ( ( *(data+23) >> 4 ) & 0x3 );
				distance[7] |= ( ( *(data+23) >> 6 ) & 0x3 );

				firmwareVersion = 0;
				firmwareVersion |= ( (*(data+24) & 1<<4 ) ? 1<<3 : 0 );
				firmwareVersion |= ( (*(data+24) & 1<<5 ) ? 1<<2 : 0 );
				firmwareVersion |= ( (*(data+24) & 1<<6 ) ? 1<<1 : 0 );
				firmwareVersion |= ( (*(data+24) & 1<<7 ) ? 1 : 0 );

				avint[0] = *(data+25);
				if( ( *(data+24) & 0x1 ) == 0 )
				{
					avint[0] = -avint[0];
				}

				uint8p = reinterpret_cast<unsigned char*>( &actualPosition[0] );
				*uint8p++ = *(data+26);
				*uint8p++ = *(data+27);
				*uint8p++ = *(data+28);
				*uint8p = *(data+29);
				actualPosition[0] = -actualPosition[0];

				//digital input
				dIn[0] = ( (*(data+30) & 1 ) > 0 );
				dIn[1] = ( (*(data+30) & 1<<1 ) > 0 );
				dIn[2] = ( (*(data+30) & 1<<2 ) > 0 );
				dIn[3] = ( (*(data+30) & 1<<3 ) > 0 );
				bumper = ( ( *(data+30) & 1<<4 ) > 0 );

				sequenceNumber = *reinterpret_cast<const unsigned int*>( data+31 );

				//slave 1 start at byte 35
				rawMotorCurrent[1] = ( *(data+35) << 2);
				distance[4] = ( *(data+36) << 2);
				distance[3] = ( *(data+37) << 2);
				ad[4] = ( *(data+38) << 2);
				ad[5] = ( *(data+39) << 2);
				ad[6] = ( *(data+40) << 2);
				ad[7] = ( *(data+41) << 2);
				distance[2] = ( *(data+42) << 2);

				rawMotorCurrent[1] |= ( *(data+43) & 0x3 );
				distance[4] |= ( ( *(data+43) >> 2 ) & 0x3 );
				distance[3] |= ( ( *(data+43) >> 4 ) & 0x3 );
				ad[4] |= ( ( *(data+43) >> 6 ) & 0x3 );

				ad[5] |= ( ( *(data+44) ) & 0x3 );
				ad[6] |= ( ( *(data+44) >> 2 ) & 0x3 );
				ad[7] |= ( ( *(data+44) >> 4 ) & 0x3 );
				distance[2] |= ( ( *(data+44) >> 6 ) & 0x3 );

				avint[1] = *(data+46);
				if( ( *(data+45) & 0x1 ) == 0 )
				{
					avint[1] = -avint[1];
				}

				uint8p = reinterpret_cast<unsigned char*>( &actualPosition[1] );
				*uint8p++ = *(data+47);
				*uint8p++ = *(data+48);
				*uint8p++ = *(data+49);
				*uint8p = *(data+50);
				actualPosition[1] = -actualPosition[1];

				dIn[4] = ( (*(data+51) & 1 ) > 0 );
				dIn[5] = ( (*(data+51) & 1<<1 ) > 0 );
				dIn[6] = ( (*(data+51) & 1<<2 ) > 0 );
				dIn[7] = ( (*(data+51) & 1<<3 ) > 0 );

				//slave 2 start at byte 56
				rawMotorCurrent[2] = ( *(data+56) << 2);
				distance[1] = ( *(data+57) << 2);

				//AD2
				distance[0] = ( *(data+58) << 2);

				distance[8] = ( *(data+63) << 2);


				rawMotorCurrent[2] |= ( *(data+64) & 0x3 );
				distance[1] |= ( ( *(data+64) >> 2 ) & 0x3 );
				distance[0] |= ( ( *(data+64) >> 4 ) & 0x3 );

				distance[8] |= ( ( *(data+65) >> 6 ) & 0x3 );

				avint[2] = *(data+67);
				if( ( *(data+66) & 0x1 ) == 0 )
				{
					avint[2] = -avint[2];
				}

				uint8p = reinterpret_cast<unsigned char*>( &actualPosition[2] );
				*uint8p++ = *(data+68);
				*uint8p++ = *(data+69);
				*uint8p++ = *(data+70);
				*uint8p = *(data+71);
				actualPosition[2] = -actualPosition[2];

				//slave 3 start at byte 77
				powerOutputRawCurrent = ( *(data+77) << 2);
				powerOutputRawCurrent |= ( *(data+85) & 0x3 );

				powerOutputCurrent = static_cast<float>( powerOutputRawCurrent ) / 156.0f;


				encoderInputVelocity = *(data+88);
				if( ( *(data+87) & 0x1 ) == 0 )
				{
					encoderInputVelocity = -encoderInputVelocity;
				}

				uint8p = reinterpret_cast<unsigned char*>( &encoderInputPosition );
				*uint8p++ = *(data+89);
				*uint8p++ = *(data+90);
				*uint8p++ = *(data+91);
				*uint8p = *(data+92);
				encoderInputPosition = -encoderInputPosition;

				for( unsigned int i=0; i<9; ++i )
				{
					distanceSensor[i] = 2.55f / 1024.0f * distance[i];
				}

				for( unsigned int i=0; i<8; ++i )
				{
					aIn[i] = 10.0f / 1024.0f * ad[i];
				}

				for( unsigned int i=0; i<3; ++i )
				{
					actualVelocity[i] = 27.0f * avint[i]; // 900 * 60 / 2000 = 27

					motorCurrent[i] = static_cast<float>( rawMotorCurrent[i] ) / 156.0f;
				}

				voltage = static_cast<float>( voltageint ) / 36.10f;
				current = static_cast<float>( currentint ) / 61.44f;

				return true;
			}

			inline void SensorState::reset()
			{
				//Set default values
				for( int mi = 0; mi < 3; mi++ )
				{
					actualVelocity[ mi ] = 0.0f;
					actualPosition[ mi ] = 0;
					motorCurrent[ mi ] = 0.0f;
					rawMotorCurrent[ mi ] = 0;
				}

				for( int i = 0; i < 8; i++)
				{
					dIn[ i ] = false;
				}

				for( int i = 0; i < 8; i++)
				{
					aIn[ i ] = 0.0f;
				}

				for( int i = 0; i < 9; i++)
				{
					distanceSensor[ i ] = 0.0f;
				}

				bumper = false;

				current = 0.0f;
				voltage = 0.0f;

				isPassiveMode = false;

				powerOutputCurrent = 0.0f;
				powerOutputRawCurrent = 0;

				encoderInputPosition = 0;
				encoderInputVelocity = 0;

				isGripperOpened = false;
				isGripperClosed = false;

				odometryX = 0.0f;
				odometryY = 0.0f;
				odometryPhi = 0.0f;

				serialLineUpdateFreqeuncy = 0;

				firmwareVersion = 0;

				sequenceNumber = 0;

				nstar_sequenceNumber = 0;
				nstar_roomId = 0;
				nstar_numSpotsVisible = 0;
				nstar_posX = 0.0f;
				nstar_posY = 0.0f;
				nstar_posTheta = 0.0f;
				nstar_magSpot0 = 0;
				nstar_magSpot1 = 0;

				std::memset( (void*)firmware_version, 0, 3 );
			}

			inline void SensorState::copy( const SensorState& other )
			{
				for( int mi = 0; mi < 3; mi++ )
				{
					actualVelocity[ mi ] = other.actualVelocity[ mi ];
					actualPosition[ mi ] = other.actualPosition[ mi ];
					motorCurrent[ mi ] = other.motorCurrent[ mi ];
					rawMotorCurrent[ mi ] = other.rawMotorCurrent[ mi ];
				}

				for( int i = 0; i < 8; i++)
				{
					dIn[ i ] = other.dIn[ i ];
				}

				for( int i = 0; i < 8; i++)
				{
					aIn[ i ] = other.aIn[ i ];
				}

				for( int i = 0; i < 9; i++)
				{
					distanceSensor[ i ] = other.distanceSensor[ i ];
				}

				bumper = other.bumper;

				current = other.current;
				voltage = other.voltage;

				isPassiveMode = other.isPassiveMode;

				powerOutputCurrent = other.powerOutputCurrent;
				powerOutputRawCurrent = other.powerOutputRawCurrent;

				encoderInputPosition = other.encoderInputPosition;
				encoderInputVelocity = other.encoderInputVelocity;

				isGripperOpened = other.isGripperOpened;
				isGripperClosed = other.isGripperClosed;

				odometryX = other.odometryX;
				odometryY = other.odometryY;
				odometryPhi = other.odometryPhi;

				serialLineUpdateFreqeuncy = other.serialLineUpdateFreqeuncy;

				firmwareVersion = other.firmwareVersion;

				sequenceNumber = other.sequenceNumber;

				nstar_sequenceNumber = other.nstar_sequenceNumber;
				nstar_roomId = other.nstar_roomId;
				nstar_numSpotsVisible = other.nstar_numSpotsVisible;
				nstar_posX = other.nstar_posX;
				nstar_posY = other.nstar_posY;
				nstar_posTheta = other.nstar_posTheta;
				nstar_magSpot0 = other.nstar_magSpot0;
				nstar_magSpot1 = other.nstar_magSpot1;

				std::memcpy( (void*)firmware_version, (const void*)other.firmware_version, 3 );
			}
		}
	}
}

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(rec::iocontrol::remotestate::SensorState)
#endif

#endif

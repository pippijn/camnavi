//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/iocontrol/robotstate/State.h"

#include "rec/iocontrol/robotstate/Encoder.h"
#include "rec/iocontrol/robotstate/Decoder.h"

#include <cmath>
#include <iostream>
#include <string.h>

using namespace rec::iocontrol::robotstate;

#define PI 3.14159265358979323846

void State::reset()
{
	Encoder* enc = new Encoder( this );
	enc->reset();
	delete enc;

	freq = 1;

	p2qUpdateCounter = 0;
	q2pUpdateCounter = 0;

	sleepVoltage = 22.5f;
	shutdownVoltage = 21.0f;

	northStarOrientation = 1;

	webcontrolLinearIncrement = 50.0f;
	webcontrolRotationIncrement = 20.0f;
	webcontrolTimeout = 10000;

	system = VIA;

	strcpy( iodVersion, "na" );
	strcpy( camdVersion, "na" );
	strcpy( controldVersion, "na" );
	strcpy( statedVersion, "na" );
	strcpy( lcddVersion, "na" );
	strcpy( webserverdVersion, "na" );
	strcpy( operatingSystemVersion, "na" );
	strcpy( gyrodVersion, "na" );
	strcpy( nstardVersion, "na" );

	strcpy( currentLanguage, "en" );

	currentApiVersion = 1;

	meanBatteryVoltage = 0.0f;
	isVoltageLow = false;

	//Odometry
	odoX = 0.0;
	odoY = 0.0;
	odoPhi = 0.0;

	isDirectControl = false;

	isIodConnected = false;

	isRvwProgramExecutionRestricted = true;

	_serialPortsInUse = 0;
}

void State::getOdometry( double* posX, double* posY, double* phi )
{
	*posX = odoX;
	*posY = odoY;
	*phi = odoPhi / PI * 180.0;
}

void State::setOdometry( double posX, double posY, double phi )
{
	odoX = posX;
	odoY = posY;
	odoPhi = phi / 180.0 * PI;

	gyro.origin = gyro.angle - odoPhi;
}

void State::refreshOdometry( float elapsedTime )
{
	//Get the actual velocity of the robot
	float vX, vY, vOmega;

	Decoder dec( this );
	dec.actualVelocity( &vX, &vY, &vOmega );

	//std::cout << vX << " " << vY << " " << vOmega << std::endl;

	//Compute the estimated change of the position and rotation
	double elapsedSeconds = (double)elapsedTime / 1000.0;
	double deltaXLocal = (double)vX     * elapsedSeconds * odometry.correctX;
	double deltaYLocal = (double)vY     * elapsedSeconds * odometry.correctY;
	double deltaOmega  = (double)vOmega * elapsedSeconds * PI / 180.0 * odometry.correctPhi;

	//Project the local coordinates to global coordinates
	if( rec::serialport::UNDEFINED != gyro.port )
	{
		odoPhi = gyro.angle - gyro.origin;
	}
	else
	{
		odoPhi += deltaOmega;
	}
	odoX += cos( odoPhi ) * deltaXLocal - sin( odoPhi ) * deltaYLocal;
	odoY += sin( odoPhi ) * deltaXLocal + cos( odoPhi ) * deltaYLocal;

	while( odoPhi >= PI )
	{
		odoPhi -= 2.0 * PI;
	}
	while( odoPhi < -PI )
	{
		odoPhi += 2.0 * PI;
	}

	//std::cout << odoX << " " << odoY << " " << odoPhi << std::endl;
}

void State::setSerialPortUsed( rec::serialport::Port port, bool isUsed )
{
	if( isUsed )
	{
		_serialPortsInUse |= ( 1 << (int)port );
	}
	else
	{
		_serialPortsInUse &= ~( 1 << (int)port );
	}
}

bool State::isSerialPortUsed( rec::serialport::Port port ) const
{
	return ( ( _serialPortsInUse & ( 1 << (int)port ) ) > 0 );
}
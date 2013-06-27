//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/iocontrol/robotstate/Encoder.h"

#include "rec/iocontrol/sercom/qdsa_encoder.h"

#include <sstream>
#include <cmath>
#include <stdlib.h>

using namespace rec::iocontrol::robotstate;

Encoder::Encoder( State* state )
: _state( state )
, _enc( new QDSA_Encoder( (unsigned char*)state->p2qfull_buffer, NB_P2Q_FULL, (unsigned char*)state->q2pfull_buffer, NB_Q2P_FULL ) )
{
}

Encoder::~Encoder()
{
	delete _enc;
}

void Encoder::reset()
{
	_enc->reset();
	_state->p2qUpdateCounter++;
}

void Encoder::setVelocity_i( unsigned int motor, float rpm )
{
	double speed = (double)rpm * _state->driveLayout.mer / _state->driveLayout.fctrl / 60.0;

	_enc->set_DV( motor, static_cast<boost::int16_t>( speed ) );
	_enc->set_Brake( motor, false );
}

void Encoder::setVelocity( unsigned int motor, float rpm )
{
	setVelocity_i( motor, rpm );
	_state->p2qUpdateCounter++;
}


void Encoder::setPowerOutputSetPoint( float setPoint )
{
	if( setPoint > 100.0f )
	{
		setPoint = 100.0f;
	}
	else if( setPoint < -100.0f )
	{
		setPoint = -100.0f;
	}

	_enc->set_DV( QDSA_Encoder::powerOutput, static_cast<boost::int16_t>( 2.55f * setPoint ) );
	_enc->set_Brake( QDSA_Encoder::powerOutput, false );
}

void Encoder::setVelocity_i( float vX, float vY, float vOmega )
{
	float m[3];
	
	projectVelocity( &m[0], &m[1], &m[2], vX, vY, vOmega, _state->driveLayout );

	for( unsigned int i=0; i<3; ++i )
	{
		setVelocity_i( i, m[i] );
	}
}

void Encoder::setVelocity( float vX, float vY, float vOmega )
{
	setVelocity_i( vX, vY, vOmega );
	_state->p2qUpdateCounter++;
}


void Encoder::stopMotors()
{
	for( unsigned int i=0; i<3; ++i )
	{
		_enc->set_DV( i, 0 );
		_enc->set_Brake( i, true );
	}

	_enc->set_DV( QDSA_Encoder::powerOutput, 0 );
	_enc->set_Brake( QDSA_Encoder::powerOutput, true );
}

void Encoder::setDigitalOutput( unsigned int i, bool on )
{
	std::ostringstream os;
	os << "DO" << i;
	_enc->set_DO( os.str(), ( on ? 1 : 0 ) );
	_state->p2qUpdateCounter++;
}

void Encoder::setBrake( unsigned int motor, bool on )
{
	_enc->set_Brake( motor, on );
	_state->p2qUpdateCounter++;
}

void Encoder::setRelay( unsigned int relay, bool on )
{
	std::ostringstream os;
	os << "R" << relay;
	_enc->set_DO( os.str(), ( on ? 1 : 0 ) );
	_state->p2qUpdateCounter++;
}

void Encoder::setShutdown()
{
	_enc->set_Shutdown( true );
	_state->p2qUpdateCounter++;
}


void Encoder::resetPosition( unsigned int motor )
{
	_enc->set_ResetPosition( motor, true );
	_state->p2qUpdateCounter++;
}


void Encoder::setPID( unsigned int motor, float kp, float ki, float kd )
{
	_enc->set_KP( motor, (boost::uint8_t)(255.0f * kp) );
	_enc->set_KI( motor, (boost::uint8_t)(255.0f * ki) );
	_enc->set_KD( motor, (boost::uint8_t)(255.0f * kd) );
	_state->p2qUpdateCounter++;
}

void Encoder::projectVelocity( float* m1, float* m2, float* m3, float vx, float vy, float omega, const DriveLayout& layout )
{
  static const double PI = 3.14159265358979323846;

  //Projection matrix
  static const double v0[2] = { -0.5 * sqrt( 3.0 ),  0.5 };
  static const double v1[2] = {  0.0              , -1.0 };
  static const double v2[2] = {  0.5 * sqrt( 3.0 ),  0.5 };

  //Scale omega with the radius of the robot
  double vOmegaScaled = 2.0 * PI * layout.rb * (double)omega / 360.0 ;

  //Convert from mm/s to RPM
  const double k = 60.0 * layout.gear / ( 2.0 * PI * layout.rw );

  //Compute the desired velocity
  *m1 = static_cast<float>( ( v0[0] * (double)vx + v0[1] * (double)vy + vOmegaScaled ) * k );
  *m2 = static_cast<float>( ( v1[0] * (double)vx + v1[1] * (double)vy + vOmegaScaled ) * k );
  *m3 = static_cast<float>( ( v2[0] * (double)vx + v2[1] * (double)vy + vOmegaScaled ) * k );
}

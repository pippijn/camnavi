//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/iocontrol/robotstate/Decoder.h"
#include "rec/iocontrol/sercom/qdsa_encoder.h"

#include <iostream>
#include <sstream>
#include <cmath>

using namespace rec::iocontrol::robotstate;

const float Decoder::batteryVoltageScaleConstant = SCALE_VOLTAGE24;

Decoder::Decoder( State* state )
: _state( state )
, _dec( new QDSA_Encoder( (unsigned char*)state->p2qfull_buffer, NB_P2Q_FULL, (unsigned char*)state->q2pfull_buffer, NB_Q2P_FULL ) )
{
}

Decoder::~Decoder()
{
	delete _dec;
}

void Decoder::actualVelocity( float* vX, float* vY, float* vOmega ) const
{
  unprojectVelocity( vX, vY, vOmega, actualVelocity( 0 ), actualVelocity( 1 ), actualVelocity( 2 ), _state->driveLayout );
}

float Decoder::actualVelocity( unsigned int motor ) const
{
	float rpm = static_cast<float>( _dec->get_AV( motor ) )* _state->driveLayout.fctrl * 60.0f / _state->driveLayout.mer; 

	return rpm;
}

float Decoder::setPointSpeed( unsigned int motor ) const
{
	boost::int16_t dv = _dec->get_DV( motor );

	//speed in rpm
	float speed = (float)dv / _state->driveLayout.mer * _state->driveLayout.fctrl * 60.0;

	return speed;
}

int Decoder::powerOutputSetPoint() const
{
	return _dec->get_DV( QDSA_Encoder::powerOutput );
}

float Decoder::motorCurrent( unsigned int motor ) const
{
	std::ostringstream os;
	os << "S" << motor << "_AD0";
	return _dec->get_AD( os.str() );
}

float Decoder::powerOutputCurrent() const
{
	std::ostringstream os;
	os << "S" << QDSA_Encoder::powerOutput << "_AD0";
	return _dec->get_AD( os.str() );
}

boost::int32_t Decoder::actualPosition( unsigned int motor ) const
{
	return _dec->get_AP( motor );
}

float Decoder::analogInput( unsigned int i ) const
{
	std::ostringstream os;
  os << "AIN" << i;
  return 10.0f / 1024.0f * (float)_dec->get_AD( os.str() );
}

bool Decoder::digitalInput( unsigned int i ) const
{
	std::ostringstream os;
	os << "DI" << i;
	return( _dec->get_DI( os.str() ) != 0 );
}

float Decoder::distance( unsigned int i ) const
{
	std::ostringstream os;
  os << "IR" << ( i + 1 );
  return 2.55f / 1024.0f * (float)_dec->get_AD( os.str() );
}

float Decoder::batteryVoltage( unsigned int* rawValue ) const
{
	unsigned int raw =  _dec->get_AD( "M_AD2" );
	if( rawValue )
	{
		*rawValue = raw;
	}
  return static_cast<float>( raw ) / SCALE_VOLTAGE24;
}

float Decoder::batteryCurrent() const
{
  return static_cast<float>( _dec->get_AD( "M_AD3" ) ) / SCALE_CURRENT;
}

bool Decoder::bumper() const
{
  return ( _dec->get_DI( "Bumper" ) != 0 );
}

std::string Decoder::firmwareVersion() const
{
	std::ostringstream os;
	os << _dec->get_FirmwareVersion();
  return os.str();
}

void Decoder::getPID( unsigned int motor, float* kp, float* ki, float* kd ) const
{
	*kp = static_cast<float>( _dec->get_KP( motor ) ) / 255.0f;
	*ki = static_cast<float>( _dec->get_KI( motor ) ) / 255.0f;
	*kd = static_cast<float>( _dec->get_KD( motor ) ) / 255.0f;
}

void Decoder::unprojectVelocity( float* vx, float* vy, float* omega, float m1, float m2, float m3, const DriveLayout& layout )
{
  static const double PI = 3.14159265358979323846;

	//std::cout << m1 << " " << m2 << " " << m3 << std::endl;

	//Convert from RPM to mm/s
  const double k = 60.0 * layout.gear / ( 2.0 * PI * layout.rw );

	*vx = static_cast<float>( ( (double)m3 - (double)m1 ) / sqrt( 3.0 ) / k );
  *vy = static_cast<float>( 2.0 / 3.0 * ( (double)m1 + 0.5 * ( (double)m3 - (double)m1 ) - (double)m2 ) / k );

	double vw = (double)*vy + (double)m2 / k;

	*omega = static_cast<float>( vw * 180.0 / ( PI * layout.rb ) );
}

//_vx = ( _v2 - _v0 ) / sqrt( 3.0f ) / k;
//  _vy = 2.0f / 3.0f * ( _v0 + 0.5f * ( _v2 - _v0 ) - _v1 ) / k;
//  vw = _vy + _v1 / k;
//
//  _omega = vw * 360.0f / ( 2.0f *pi * rb );
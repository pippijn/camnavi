//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/nstar/ComImpl.h"
#include "rec/nstar/Com.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>

#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

void msleep( unsigned int ms )
{
#ifdef WIN32
	SleepEx( ms, false );
#else
	::usleep( ms * 1000 );
#endif
}

//namespace rec
//{
//	namespace nstar
//	{
//		static int _argc = 1;
//		static char* _argv[] = {"a","b"};
//	}
//}

using namespace rec::nstar;

const float ComImpl::ceilingCalConversionFactor = 18918.0f;

unsigned char ComImpl::checksum( const unsigned char* buffer, unsigned int bufferSize ) const
{
	unsigned char chks = 0;

	for( unsigned int i = 0; i < bufferSize; ++i )
	{
		chks += buffer[i];
	}

	return ( 0xff - chks );
}

bool ComImpl::isPacketCorrect( const unsigned char *packet, unsigned int len)
{
    /* check for non zero length */
    if (len == 0)
    {
		reportError( "The packet length is zero" );
        return false;
    }
        
    /* check for status */
    if (packet[0] != 0x00)
    {
		std::ostringstream os;
		os << "Error receving correct packet. Status: " << (int)packet[0];
		reportError( os.str().c_str() );
        return false;
    }

    /* check for correct length */
    if (packet[1] != len)
    {
		std::ostringstream os;
		os << "Length of packet " << (int)packet[1] << " doesnt match the value of the length " << len << " passed to the function";
		reportError( os.str().c_str() );
        return false;
    }

    /* checksum */
    unsigned char sum = 0;
    for (unsigned int i = 0; i < len; i++)
    {
        sum+=packet[i];
    }   

    if ((sum % 255) != 0)
    {
		reportError( "Checksum error" );
        return false;
    }

    return true;
}

float ComImpl::convert_to_float(const unsigned char *buffer, unsigned int input_pos)
{   
    float f;
    unsigned char *p = (unsigned char *)&f;
    p[0] = buffer[input_pos];
    p[1] = buffer[input_pos+1];
    p[2] = buffer[input_pos+2];
    p[3] = buffer[input_pos+3];
    return f;
}

void ComImpl::convert_float_to_char_array(unsigned char *buf, float f)
{
    unsigned char *p = (unsigned char *)&f;
    buf[0] = p[0];
    buf[1] = p[1];
    buf[2] = p[2];
    buf[3] = p[3];

}

ComImpl::ComImpl( Com* com )
: _com( com )
{
	//if( NULL == QCoreApplication::instance() )
	//{
	//	new QCoreApplication( _argc, _argv );
	//}
	reset();
}

bool ComImpl::open( rec::nstar::Com::port_t port )
{
	if( rec::nstar::Com::UndefinedPort == port )
	{
#ifdef WIN32
		for( int i=(int)rec::serialport::COM1; i<=(int)rec::serialport::COM11; ++i )
#else
		for( int i=(int)rec::serialport::USB0; i<(int)rec::serialport::USB7; ++i )
#endif
		{
			if( open( (rec::serialport::Port)i ) )
			{
				return true;
			}
		}
	}
	else
	{
		if( open( (rec::serialport::Port)port ) )
		{
			return true;
		}
	}

	return false;
}

bool ComImpl::open( rec::serialport::Port port )
{
	static const unsigned int speedsSize = 2;
	static const unsigned int speeds[speedsSize] = { 1200, 115200 };

	for( unsigned int speed=0; speed < speedsSize; ++speed)
	{
		unsigned int state = 0;
		bool retry = true;
		try
		{
			_serialPort.close();

			_serialPort.open( port, speeds[speed], 1000 );

			state = 1;

			//std::cout << "Try to connect at " << speeds[speed] << " baud" << std::endl;
			if( false == getVersion() )
			{
				std::ostringstream os;
				os << "Failed connecting to " << rec::serialport::friendlyName( port ) << " with speed " << speeds[speed] << " baud";
				reportError( os.str().c_str() );
				continue;
			}

			state = 2;
			if( _serialPort.speed() < 115200 )
			{
				state = 3;
				setBaudRate( 115200 );
				continue;
			}

			_serialPort.setReadTimeout( 1000 );

			return setCeilingCal_i( _ceilingCal );
		}
		catch( const rec::serialport::SerialPortException& )
		{
			switch( state )
			{
			case 0: //open failed
				retry = false;
				break;

			default:
				break;
			}
		}

		if( false == retry )
		{
			break;
		}
	}

	return false;
}

bool ComImpl::isOpen() const
{
	return _serialPort.isOpen();
}


void ComImpl::close()
{
	stopContinuousReport();

	stop();

	if( _serialPort.isOpen() )
	{
		setBaudRate( 1200 );
		_serialPort.close();
	}

	reset();
}

int ComImpl::version() const
{
	return _version;
}

const char* ComImpl::portString() const
{
	_portString = rec::serialport::friendlyName( _serialPort.port() );
	return _portString.c_str();
}

unsigned int ComImpl::speed() const
{
	return _serialPort.speed();
}

bool ComImpl::setReportFlags(
							 bool report_pose,
							 unsigned int spotReportMask,
							 unsigned int magnitudeReportMask,
							 unsigned int spot_avail_threshold )
{
	if( !isOpen() )
	{
		return false;
	}

	_spotpos_id.clear();
	for( int i=0; i<MAX_NUM_SPOTS; ++i )
	{
		if( spotReportMask & ( 1 << i ) )
		{
			_spotpos_id.push_back( i );
		}
	}

	_magnitude_id.clear();
	for( int i=0; i<MAX_NUM_SPOTS; ++i )
	{
		if( magnitudeReportMask & ( 1 << i ) )
		{
			_magnitude_id.push_back( i );
		}
	}

	unsigned char buf[12];

	/* Report flag variables */
	unsigned char spot_report_mask[3];
	unsigned char mag_report_mask[3];
	unsigned char spot_avail_thresh[2];

	spot_report_mask[0] = (unsigned char) (spotReportMask & 0x000000ff );
	spot_report_mask[1] = (unsigned char) ( (spotReportMask >> 8) & 0x000000ff );    
	spot_report_mask[2] = (unsigned char) ( (spotReportMask >> 16) & 0x000000ff );    

	mag_report_mask[0] = (unsigned char) (magnitudeReportMask & 0x000000ff );
	mag_report_mask[1] = (unsigned char) ( (magnitudeReportMask >> 8) & 0x000000ff );    
	mag_report_mask[2] = (unsigned char) ( (magnitudeReportMask >> 16) & 0x000000ff );       

	spot_avail_thresh[0] = (unsigned char) ( 0x00ff & spot_avail_threshold );
	spot_avail_thresh[1] = (unsigned char) ( 0x00ff & (spot_avail_threshold >> 8) );        


	/* Packet to send to NS */
	buf[0] = 0x2;  // Command ID
	buf[1] = 0xc;  // Packet Length 
	buf[2] = 0x00; // Checksum (to be computed later)
	buf[3] = report_pose;  // whether to report pose
	buf[4] = spot_report_mask[2];   // spot_report_mask, bits 17-24
	buf[5] = spot_report_mask[1];   // spot_report_mask, bits 9-16
	buf[6] = spot_report_mask[0];   // spot_report_mask, bits 1-8
	buf[7] = mag_report_mask[2];  // mag_report_mask, bits 17-24
	buf[8] = mag_report_mask[1];  // mag_report_mask, bits 9-16
	buf[9] = mag_report_mask[0];  // mag_report_mask, bits 1-8
	buf[10]= spot_avail_thresh[1];  // spot_avail_thresh, bits 9-16
	buf[11]= spot_avail_thresh[0];  // spot_avail_thresh, bits 1-8

	buf[2] = checksum(buf, 12); // Checksum

	int bytes_written = _serialPort.write( buf, 12 );

	//msleep( 200 );

	if( 12 != bytes_written )
	{
		reportError( "Error sending report flags" );
		return false;
	}

	int bytes_read = _serialPort.read( buf, 3 );

	if( 3 != bytes_read )
	{
		reportError( "Error reading response to sending report flags" );
		return false;
	}

	if( false == isPacketCorrect( buf, 3 ) )
	{
		reportError( "Error in packet received" );
		return false;
	}

	return true;
}

void ComImpl::setRoomAndCeilingCal( unsigned int room, float ceilingCal )
{
	bool restart = false;
	bool roomChanged = false;
	bool ceilingCalChanged = false;

	if( room > 8 )
	{
		return;
	}

	if( room != _room )
	{
		roomChanged = true;
	}
	
	if( ceilingCal != _ceilingCal )
	{
		ceilingCalChanged = true;
	}

	if( ( false == roomChanged ) && ( false == ceilingCalChanged ) )
	{
		return;
	}

	if( isRunning() )
	{
		restart = true;
		stopContinuousReport();
	}

	_room = room;
	_ceilingCal = ceilingCal;

	if( !isOpen() )
	{
		return;
	}

	setCeilingCal_i( ceilingCal );

	if( restart )
	{
		startContinuousReport();
	}
}

void ComImpl::setRoom( unsigned int room )
{
	if( ( room == _room ) || ( room > 8 ) )
	{
		return;
	}

	_room = room;

	if( isRunning() )
	{
		stopContinuousReport();
		startContinuousReport();
	}
}

bool ComImpl::setCeilingCal( float ceilingCal )
{
	if( ceilingCal == _ceilingCal )
	{
		return true;
	}

	_ceilingCal = ceilingCal;

	if( !isOpen() )
	{
		return true;
	}

	bool restart = false;

	if( isRunning() )
	{
		restart = true;
		stopContinuousReport();
	}

	bool ret = setCeilingCal_i( ceilingCal );

	if( restart )
	{
		startContinuousReport();
	}

	return ret;
}

bool ComImpl::setCeilingCal_i( float ceilingCal )
{
	if( _version < 200 ) //command unknown by old NS detector
	{
		//std::cout << "This is an old NS detector. We can not set ceilingCal." << std::endl;
		return true;
	}

	//std::cout << "Setting ceilingCal to " << ceilingCal << std::endl;

	unsigned char buf[12];

	/* Packet to send to NS */
	buf[0] = 0x0b;  // Command ID
	buf[1] = 0x0c;  // Packet Length 
	buf[2] = 0x00; // Checksum (to be computed later)
	buf[3] = _room;  // whether to report pose

	float a = ceilingCal / ceilingCalConversionFactor;

	convert_float_to_char_array( buf+4, a );
	convert_float_to_char_array( buf+8, a );

	buf[2] = checksum(buf, 12); // Checksum

	int bytes_written = _serialPort.write( buf, 12 );

	if( 12 != bytes_written )
	{
		reportError( "Error sending set ceiling cal" );
		return false;
	}

	int bytes_read = _serialPort.read( buf, 3 );

	if( 3 != bytes_read )
	{
		reportError( "Error reading response to sending ceiling cal" );
		return false;
	}

	if( false == isPacketCorrect( buf, 3 ) )
	{
		reportError( "Error in packet received" );
		return false;
	}

	return true;
}

float ComImpl::ceilingCal()
{
	if( !isOpen() )
	{
		return 0.0f;
	}

	bool restart = false;

	if( isRunning() )
	{
		restart = true;
		stopContinuousReport();
	}

	unsigned char buf[11];

	/* Packet to send to NS */
	buf[0] = 0x0a;  // Command ID
	buf[1] = 0x04;  // Packet Length 
	buf[2] = 0x00; // Checksum (to be computed later)
	buf[3] = _room;  // room we request ceiling cal for

	buf[2] = checksum(buf, 4); // Checksum

	int bytes_written = _serialPort.write( buf, 4 );

	if( 4 != bytes_written )
	{
		reportError( "Error sending get ceiling cal" );
		return 0;
	}

	int bytes_read = _serialPort.read( buf, 11 );

	if( 11 != bytes_read )
	{
		reportError( "Error reading response to get ceiling cal" );
		return 0;
	}

	if( false == isPacketCorrect( buf, 11 ) )
	{
		reportError( "Error in packet received" );
		return 0;
	}

	float a = convert_to_float( buf, 3 );
	float b = convert_to_float( buf, 7 );

	if( a != b )
	{
		reportError( "Ceiling cal differs" );
	}

	if( restart )
	{
		startContinuousReport();
	}

	return a * ceilingCalConversionFactor;
}

bool ComImpl::startContinuousReport()
{
	if( !isOpen() )
	{
		return false;
	}

	checkSpotMask();

	unsigned char buf[4];
    int bytes_written;

    /* Packet to send the continuous report */
    buf[0] = 0x04;        // Command ID
    buf[1] = 0x04;        // Packet length
    buf[2] = 0x00;      // Checksum
    buf[3] = _room; // Rooom ID (hence pair of spots) used to calculate pose
    
    buf[2] = checksum(buf, 4);

	bytes_written = _serialPort.write( buf, 4 );

	//msleep( 200 );

	if( 4 != bytes_written )
    {
		reportError( "Could not send continous report packet." );
        return false;
    }

	start();

	return true;
}

bool ComImpl::stopContinuousReport()
{
	if( !isOpen() )
	{
		return false;
	}

	unsigned char buf[3];

	buf[0] = 0x05;  // Command ID 
    buf[1] = 0x03;  // Packet length 
    buf[2] = 0xf7;  // Checksum

	int bytes_written = _serialPort.write( buf, 3 );

	if( 3 != bytes_written )
    {
		reportError( "Could not send stop report packet." );
		stop();
        return false;
    }

	if( false == wait( 1000 ) )
	{
		stop();
	}

	return true;
}

bool ComImpl::singleReport()
{
	if( !isOpen() )
	{
		return false;
	}

	checkSpotMask();

	unsigned char buf[4];
    int bytes_written;

    /* Packet to send the continuous report */
    buf[0] = 0x03;        // Command ID
    buf[1] = 0x04;        // Packet length
    buf[2] = 0x00;      // Checksum
    buf[3] = _room; // Rooom ID (hence pair of spots) used to calculate pose
    
    buf[2] = checksum(buf, 4);

	bytes_written = _serialPort.write( buf, 4 );

	//msleep( 200 );

	if( 4 != bytes_written )
    {
		reportError( "Could not send single report packet." );
        return false;
    }

	_singleReportInProgress = true;

	for( int i=0; i<5; ++i )
	{
		if( false == receivePacket() )
		{
			reportError( "Error receiving answer to single report" );
			return false;
		}

		if( false == _singleReportInProgress )
		{
			break;
		}
	}

	if( true == _singleReportInProgress )
	{
		reportError( "missed report end after single report" );
		return false;
	}

	return true;
}

bool ComImpl::getVersion()
{
	unsigned char getVersion[3] = { 0x00, 0x03, 0xfc };

	_serialPort.write( getVersion, 3 );

	//msleep( 200 );

	unsigned char buf[4];
	memset( buf, 0, 4 );

	int bytes_read = _serialPort.read( buf, 4 );
	if( 4 != bytes_read )
	{
		reportError( "Error reading 4 bytes as answer to get version" );
		return false;
	}

	if( false == isPacketCorrect( buf, 4 ) )
	{
		return false;
	}

	_version = buf[3];

	return true;
}

bool ComImpl::setBaudRate( unsigned int speed )
{
	unsigned char speed_num = 0;

	switch( speed )
	{
	case 1200:
		speed_num = 0;
		break;

	case 19200:
		speed_num = 1;
		break;

	case 57600:
		speed_num = 2;
		break;

	default:
		speed_num = 3;
		break;
	}

	unsigned char buf[4];

	buf[0] = 0x09;
	buf[1] = 0x04;
	buf[2] = 0x00;
	buf[3] = speed_num;

	buf[2] = checksum( buf, 4 );

	if( 4 != _serialPort.write( buf, 4 ) )
	{
		reportError( "Error writing new baudrate" );
		return false;
	}

	msleep( 500 );

	return true;
}

void ComImpl::run()
{
	while( _run )
	{
		if( false == receivePacket() )
		{
			_com->continuousReportErrorEvent();
			break;
		}
	}
}

void ComImpl::checkSpotMask()
{
	bool spotsInReport = false;
	std::vector<int>::const_iterator iter = std::find( _spotpos_id.begin(), _spotpos_id.end(), _room * 2 );
	if( _spotpos_id.end() != iter )
	{
		++iter;
		if( _spotpos_id.end() != iter )
		{
			if( *iter == _room * 2 + 1 )
			{
				spotsInReport = true;
			}
		}
	}

	if( false == spotsInReport )
	{
		//std::cout << "Spots for room " << _room << " are not in spot_report_mask" << std::endl;
		//std::cout << "I will set the correct mask for you now" << std::endl;
		setReportFlags( true, ( 0x3 << _room * 2 ), 0, 0 );
	}
}

void ComImpl::reset()
{
	_version = -1;
	_sequenceNumber = 0;
	_room = 0;
	_ceilingCal = 1.0f;
	_spotpos_id.clear();
	_magnitude_id.clear();
}

bool ComImpl::receivePacket()
{
	unsigned char buf[256];

	if( _serialPort.read( buf, 3 ) < 3 )
	{
		reportError( "Error reading the first 3 bytes of packet" );
		return false;
	}

	if( buf[1] < 3 )
	{
		std::ostringstream os;
		os << "Invalid packet length: " << (int)buf[1];
		reportError( os.str().c_str() );
		return false;
	}

	int bytes_to_read = buf[1] - 3;

	if( _serialPort.read( (buf+3), bytes_to_read ) < bytes_to_read )
	{
		reportError( "Error reading rest of packet" );
		return false;
	}

	if( false == isPacketCorrect( buf, buf[1] ) )
	{
		reportError( "Packet has errors" );
		return false;
	}

	switch( buf[3] )
	{
		case 0x03: //spot position report
			{
				int num_spots_cont_report = _spotpos_id.size();
				if( buf[1] != 6 * num_spots_cont_report + 4 )
				{
					reportError( "spot position report has invalid length" );
					return false;
				}

				SpotPositionReport report;

				int num_bytes_parsed = 4;
				for( int i = 0; i < num_spots_cont_report; ++i )
				{
					int curr_spot_id = _spotpos_id[i];

					report.spot_x[curr_spot_id] = (short) ( (buf[num_bytes_parsed] << 8) | 
						(buf[num_bytes_parsed + 1]) );
					num_bytes_parsed += 2;

					report.spot_y[curr_spot_id] = (short) ( (buf[num_bytes_parsed] << 8) | 
						(buf[num_bytes_parsed + 1]) );
					num_bytes_parsed += 2;

					report.spot_mag[curr_spot_id] = (unsigned short)( (buf[num_bytes_parsed] << 8) | 
						(buf[num_bytes_parsed + 1]) );

					num_bytes_parsed += 2;
				}

				reportSpotPositionEvent( report );
			}
			break;

		case 0x04: //magnitude report
			{
				int num_mag_cont_report = _magnitude_id.size();
				if( buf[1] != 4 + 2 * num_mag_cont_report + 4 )
				{
					reportError( "magnitude report has invalid length" );
					return false;
				}

				MagnitudeReport report;

				int num_bytes_parsed = 4;
				for ( int i = 0; i < num_mag_cont_report; ++i )
				{
					int curr_spot_id = _magnitude_id[i];

					report.magnitude[curr_spot_id] = (unsigned short) (buf[num_bytes_parsed] << 8) | 
						(buf[num_bytes_parsed + 1]);
					num_bytes_parsed += 2;
				}

				reportMagnitudeEvent( report );
			}
			break;

		case 0x05: //pose report
			{
				PoseReport report;

				report.sequenceNumber = _sequenceNumber++;
				report.room = _room;

				if (buf[4] == 0x03) /* 2 good spots */
				{
					report.numGoodSpots = 2;
					report.pose_x = (short) (buf[5] << 8) | (buf[6]); /* x */
					report.pose_y = (short) (buf[7] << 8) | (buf[8]); /* y */
				}
				else /* only one good spot */
				{
					report.numGoodSpots = 1;
					/* NOTE: bytes 7-8 are ignored */
					report.r = (buf[5] << 8) | (buf[6]); /* R */
				}

				//buf[9] = 0xed;
				//buf[10] = 0x6a;

				signed short theta = ( ( buf[9] << 8 ) | buf[10] );
				report.pose_theta = (float)theta / (float)(1<<13); /* theta */

				report.spot0_mag = (unsigned short) (buf[11] << 8) | (buf[12]); /* spot0 magnitude */
				report.spot1_mag = (unsigned short) (buf[13] << 8) | (buf[14]); /* spot1 magnitude */

				reportPoseEvent( report );
			}
			break;

		case 0x06: //report end
			{
				if( 0x03 != buf[4] )
				{
					//no more packets coming
					reportEndEvent();
				}
			}
			break;

		default:
			reportError( "Received unhandled packet" );
			return false;
	}

	return true;
}

void ComImpl::reportPoseEvent( const PoseReport& report )
{
	_com->reportPoseEvent( report );
}

void ComImpl::reportSpotPositionEvent( const SpotPositionReport& report )
{
	_com->reportSpotPositionEvent( report );
}

void ComImpl::reportMagnitudeEvent( const MagnitudeReport& report )
{
	_com->reportMagnitudeEvent( report );
}

void ComImpl::reportEndEvent()
{
	_singleReportInProgress = false;
	signalStop();
}

void ComImpl::reportError( const char* error )
{
	_com->reportError( error );
}

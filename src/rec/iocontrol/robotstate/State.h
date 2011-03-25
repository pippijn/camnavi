//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTSTATE_STATE_H_
#define _REC_ROBOTSTATE_STATE_H_

#include <string>
#include <map>

#include "rec/iocontrol/sercom/qdsa_protocol.h"
#include "rec/serialport/Port.h"

namespace rec
{
	namespace iocontrol
	{
		namespace robotstate
		{
			/**
			* @brief	Holds the drive layout of the Robot. 
			*/
			class DriveLayout
			{
			public:
				DriveLayout()
					: rb( 125.0 )
					, rw( 40.0 )
					, fctrl( 900.0 )
					, gear( 16.0 )
					,	mer( 2000.0 )
				{
				}

				/**
				* distance from center to wheel center
				* Set by stated from robotino.xml.
				*/
				double rb;

				/**
				* radius wheel in mm
				* Set by stated from robotino.xml.
				*/
				double rw;

				/**
				* frequency of control loop measuring the speed
				* Set by stated from robotino.xml.
				*/
				double fctrl;

				/**
				* gear
				* Set by stated from robotino.xml.
				*/
				double gear;

				/**
				* motor encoder resolution
				* Set by stated from robotino.xml.
				*/
				double mer;
			};

			/**
			* @brief	Holds configuration of the power output (motor 4). 
			*/
			class PowerOutputConfiguration
			{
			public:
				PowerOutputConfiguration()
					: maxCurrent( 1023 )
					, onReach( maxCurrent )
					, setPointReduction( 0.8f )
					, exceedTime( 3000 )
				{
				}

				/**
				* The output is set to this value after maxCurrent was reached.
				* This only takes effect if isRestartOnPolarityChange is true or restartTimout > 0
				*/
				unsigned int onReach;

				/**
				* The maximum current delivered by the output
				*/
				unsigned int maxCurrent;

				/**
				* If maxCurrent is reached the set point speed is reduced by multiplication with currentReduction as long as current is greater onReach
				*/
				float setPointReduction;

				/**
				* time in milliseconds the maxmimum current given by value has to be exceeded before current limitation is activated
				*/
				unsigned int exceedTime;
			};

			/**
			@brief Holds odometry correction constants
			@see State::refreshOdometry
			*/
			class Odometry
			{
			public:
				Odometry()
					: correctX( 1.3 )
					, correctY( 1.3 )
					, correctPhi( 1.0 )
				{
				}

				double correctX;
				double correctY;
				double correctPhi;
			};

			class Gyroscope
			{
			public:
				Gyroscope()
					: port( rec::serialport::UNDEFINED )
					, rate( 0.0f )
					, angle( 0.0f )
					, origin( 0.0f )
				{
				}

				//rec::serialport::Port
				int port;
				float rate; //in rad/s
				float angle; //in rad
				float origin; //in rad
			};

			class NorthStar
			{
			public:
				NorthStar()
					: port( rec::serialport::UNDEFINED )
					, pose_x( 0 )
					, pose_y( 0 )
					, r( 0 )
					, pose_theta( 0.0f )
					, numGoodSpots( 0 )
					, spot0_mag( 0 )
					, spot1_mag( 0 )
					, sequenceNumber( 0 )
					, roomReported( 0 )
					, room( 0 )
					, ceilingCal( 1.0f )
				{
				}

				//rec::serialport::Port
				int port;

				//reported from nstard
				signed short pose_x;
				signed short pose_y;
				unsigned short r;
				float pose_theta;
				unsigned short numGoodSpots;
				unsigned short spot0_mag;
				unsigned short spot1_mag;
				unsigned int sequenceNumber;
				int roomReported;

				//settings for nstard
				int room;
				float ceilingCal;
			};

			/**
			@brief Holds emergency stop parameters
			*/
			class EmergencyStop
			{
			public:
				EmergencyStop()
					: isEnabled( false )
					, timeout( 5000 )
					, maxMotorSpeed( 200 )
				{
				}

				/**
				* If true, Robotino stops for emergencyStopTimeout milliseconds if the bumper is activated.
				* Furthermore the speed of the motors is limited to maxMotorSpeed.
				* @see emergencyStopTimeout
				*/
				bool isEnabled;

				/**
				* The time in milliseconds Robotino stops after the bumper is activated
				* @see isEnabled
				*/
				unsigned int timeout;

				/**
				* The maximum motor speed in rpm.
				* If motorSpeed > maxMotorSpeed, motorSpeed = maxMotorSpeed
				* If motorSpeed < maxMotorSpeed, motorSpeed = -maxMotorSpeed
				* @see isEnabled
				*/
				unsigned int maxMotorSpeed;
			};

			/**
			@brief Holds data used for workspace control
			*/
			class WorkSpace
			{
			public:
				WorkSpace()
					: isEnabled( false )
					, xmin( -2.0 )
					, xmax( 2.0 )
					, ymin( -2.0 )
					, ymax( 2.0 )
					, isBreached( false )
				{
				}

				bool operator!=( const WorkSpace& other ) const
				{
					return ( ( other.xmin != xmin ) || ( other.xmax != xmax ) || ( other.ymin != ymin ) || ( other.ymax != ymax ) );
				}

				bool isEnabled;

				double xmin;
				double xmax;
				double ymin;
				double ymax;

				bool isBreached;
			};

			class EA09
			{
			public:
				EA09()
					: rSeqCount( 0 )
					, tSeqCount( 0 )
				{
				}

				unsigned int rSeqCount; //sequence count of messages received from EA09
				unsigned int tSeqCount; //sequence count of messages to be written to EA09

				unsigned char rBuf[255]; //message buffer of message received from EA09
				unsigned char tBuf[255]; //message buffer of message to be written to EA09
			};

			/**
			* @brief	Holds the state of the Robot. 
			*/
			class State
			{
			public:
				typedef enum
				{
					VIA, //Kontron MopsLcdVE PC104
					AMD, //Kontron MopsLcdLx PC104+
					SBO, //Festo SBO
					UnknownSystem
				} System_t;

				//constants
				static const int sharedMemoryKey = 5000;
				static const int versionStrLength = 20;

				/** The number of motors */
				static const unsigned int numMotors = 3;
				/** The number of digital outputs */
				static const unsigned int numDigitalOutputs = 8;
				/** The number of digital inputs */
				static const unsigned int numDigitalInputs = 8;
				/** The number of analog inputs */
				static const unsigned int numAnalogInputs = 8;
				/** The number of distance sensors */
				static const unsigned int numDistanceSensors = 9;
				/** The number of relays */
				static const int numRelays = 2;

				/**
				* Mean battery voltage calculated by controld
				*/
				float meanBatteryVoltage;

				/**
				* Indicating if the battery voltage is low.
				* Set by controld.
				*/
				bool isVoltageLow;

				/**Set by controld. Frequency of successfull reading from the serial line.*/
				unsigned int freq;

				/**Increased whenever iod writes new data into p2qfull_buffer*/
				unsigned int p2qUpdateCounter;

				/**Increased whenever rtaiserd writes new data into q2pfull_buffer*/
				unsigned int q2pUpdateCounter;

				/** Data send directly to microcontrollers if apiVersion = 1*/
				char p2qfull_buffer[NB_P2Q_FULL];

				/** Data received from microcontrollers if apiVersion = 1*/
				char q2pfull_buffer[NB_Q2P_FULL];

				/**
				* If battery voltage is lower than this value the motors stop.
				* Set by stated from robotino.xml.
				*/
				float sleepVoltage;

				/**
				* If battery voltage is lower than this value the Robotino switches itself of.
				* Set by stated from robotino.xml.
				*/
				float shutdownVoltage;

				/**
				* See documentation on the wiki.
				* Set by stated from robotino.xml.
				*/
				unsigned int northStarOrientation;

				/**
				* Increase the linear speed by this value if Robotino is control by the web interface.
				* Set by stated from robotino.xml.
				*/
				float webcontrolLinearIncrement;

				/**
				* Increase the rotation speed by this value if Robotino is control by the web interface.
				* Set by stated from robotino.xml.
				*/
				float webcontrolRotationIncrement;

				/**
				* Robotino stops after webcontrolTimeout milliseconds of inactivity of the webcontrol
				* This means you have to hit a button in the webcontrol to prevent Robotino from stopping
				*/
				unsigned int webcontrolTimeout;

				DriveLayout driveLayout;

				EmergencyStop emergencyStop;

				/**
				* The hardware we are working on.
				* Set by stated from looking at /proc/cpuinfo.
				*/
				System_t system;

				WorkSpace workSpace;

				/**
				* Null terminated string containing the active language in ISO 639-1 code
				* Examples of ISO 639-1:
				* German: de
				* English: en
				* Spanish: es
				* French: fr
				* Korean: ko
				* Japanese: ja
				*/
				char currentLanguage[3];

				/**
				* Indicating which setup is currently active
				* Possible values are:
				* 1 - stated, controld, iod, camd, lcdd
				* 2 - stated, controld, webserverd, lcdd
				*/
				unsigned int currentApiVersion;

				char iodVersion[versionStrLength];
				char camdVersion[versionStrLength];
				char controldVersion[versionStrLength];
				char statedVersion[versionStrLength];
				char lcddVersion[versionStrLength];
				char webserverdVersion[versionStrLength];
				char operatingSystemVersion[versionStrLength];
				char gyrodVersion[versionStrLength];
				char nstardVersion[versionStrLength];

				/**
				* The power output configuration.
				*/
				PowerOutputConfiguration powerOutputConfig;

				/**
				Odometry configuration.
				*/
				Odometry odometry;

				/** 
				* Reset the state.
				*/
				void reset();

				/** 
				* Get the current position of the robot
				*
				* @param posX x position of the robot in mm
				* @param posY y position of the robot in mm
				* @param phi rotation of the robot in degrees
				*/
				void getOdometry( double* posX, double* posY, double* phi );

				/** 
				* Set the current positon and rotation of the robot
				*
				* @param posX x position of the robot in mm
				* @param posY y position of the robot in mm
				* @param phi rotation of the robot in degrees
				*/
				void setOdometry( double posX, double posY, double phi );

				/** 
				* Referesh the absolute position of the robot.
				* Should be called after every change of the actual velocity of the robot
				*
				* @param elapsedTime elapsed time in milliseconds since the last call of refreshOdometry
				*/
				void refreshOdometry( float elapsedTime );

				/**
				This flag is false by default. If you have a control program directly using
				this shared memory segment to control Robotino, set this flag true.
				When true iod or webserverd will not control Robotino's actors any more but
				are still able to communicate sensor readings. If your program exits do not forget
				to set this flag to false.
				*/
				bool isDirectControl;

				/**
				This flag is set by iod if a connection (to Robotino View) is established.
				*/
				bool isIodConnected;

				/**
				If set to true, rvw programs are not executed from lcdd if they contain a camera functionblock or
				if they are a sequence control program. If set ti false all programs are executed from lcdd even though
				Robotino might hang during execution, because of hardware restrictions.
				The default is true. This can be changed in /etc/robotino/robotino.xml
				*/
				bool isRvwProgramExecutionRestricted;

				/**
				Gyroscope configuration
				*/
				Gyroscope gyro;

				/**
				NorthStar configuration
				*/
				NorthStar northStar;

				void setSerialPortUsed( rec::serialport::Port port, bool isUsed );

				bool isSerialPortUsed( rec::serialport::Port port ) const;

				EA09 ea09;

			private:
				//odometry
				/** x position in mm */
				double odoX;
				/** y position in mm */
				double odoY;
				/** rotation in rad */
				double odoPhi;

				/**
				Mask to store which serial ports are used by applications. This is used to prevent the daemons to scan a port that is already
				in use by someone else.
				@see setSerialPortUsed isSerialPortUsed
				*/
				unsigned int _serialPortsInUse;
			};
		}
	}
}

#endif

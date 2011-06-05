//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_NSTAR_COM_
#define _REC_NSTAR_COM_

#define MAX_NUM_SPOTS 20

namespace rec
{
	namespace nstar
	{
		class ComImpl;
		class PoseReport;
		class SpotPositionReport;
		class MagnitudeReport;

		class
#ifdef WIN32
#  ifdef rec_nstar_EXPORTS
		__declspec(dllexport)
#  endif
#endif
		Com
		{
		public:
			typedef enum Port { UndefinedPort, COM1, COM2, COM3, COM4, COM5, COM6, COM7, COM8, COM9, COM10, COM11, COM12, USB0, USB1, USB2, USB3, USB4, USB5, USB6, USB7 } port_t;
			
			Com();

			virtual ~Com();

			/**
			* @param port The port NS is connected to. If port is UndefinedPort this method scan the available ports for a connected NS1/2 detector.
			*/
			bool open( port_t port = UndefinedPort );

			bool isOpen() const;

			void close();

			int version() const;

			const char* portString() const;

			unsigned int speed() const;

			/**
			* The SETREPORTFLAGS command determines what information is sent by the detector when
			* it receives a REPORT or CONTINUOUSREPORT command from the host.
			* @param report_pose false = indicates that no POSE data should be returned
			*                    true = detector pose date should be returned
			* @param spot_report_mask bit field indicating which spots positions returned in the REPORT_SPOTPOS packet
			*                         0 = spot position not returned
			*                         1 = spot position returned
			*                         bit 0 refers to room 0, spot 0
			*                         bit 1 refers to room 0, spot 1
			*                         bit 2 refers to room 1, spot 0 etc
			* @param magnitude_report_mask bit field indicating which spot magnitudes returned in the REPORT_MAGNITUDES packet
			*                              0 = spot magnitude not returned
			*                              1 = spot magnitude returned
			*                              bit 0 refers to room 0, spot 0
			*                              bit 1 refers to room 0, spot 1
			*                              bit 2 refers to room 1, spot 0 etc
			* @param spot_avail_threshold a threshold value for determining if a spot is available : 16bit referring to a spot magnitude
			*                            The spot_avail_threshold is different from the threshold set using the SETTHRESHOLD command.
			*                            This threshold is only used to determine at what magnitude the detector considers a signal a real
			*                            spot and not noise. This value may be quite a bit lower than the magnitude of a spot that the
			*                            system would actually use to calculate its pose.
			*/
			bool setReportFlags(
				bool report_pose,
				unsigned int spot_report_mask,
				unsigned int magnitude_report_mask,
				unsigned int spot_avail_threshold );

			void setRoom( unsigned int room );

			/**
			Set the ceiling calibration factor.
			@param ceilingCal The distance in mm from detector to ceiling.
			*/
			bool setCeilingCal( float ceilingCal );

			float ceilingCal();

			void setRoomAndCeilingCal( unsigned int room, float ceilingCal );

			bool startContinuousReport();

			bool stopContinuousReport();

			bool isContinousReportActive() const;

			/**
			Request a single report. Stops a running continuous report first.
			* @param room The room number. Rooms are numbered starting with 0. This is different from the Northstar Tool where rooms are numbered starting with 1.
			*             If you selected room x in the Northstar Tool you have to set room to x-1.
			*             Make sure to set the spot_report_mask to contain the room's spots.
			*             Example: room=3, spot_report_mask=192
			* @see setReportFlags
			*/
			bool singleReport();

			virtual void reportPoseEvent( const PoseReport& report );

			virtual void reportSpotPositionEvent( const SpotPositionReport& report );

			virtual void reportMagnitudeEvent( const MagnitudeReport& report );

			virtual void reportError( const char* error );

			virtual void continuousReportErrorEvent();

		private:
			ComImpl* _impl;
		};

		class PoseReport
		{
		public:
			PoseReport()
				: pose_x( 0 )
				, pose_y( 0 )
				, r( 0 )
				, pose_theta( 0.0f )
				, numGoodSpots( 0 )
				, spot0_mag( 0 )
				, spot1_mag( 0 )
				, sequenceNumber( 0 )
				, room( 0 )
			{
			}

			signed short pose_x;
			signed short pose_y;
			unsigned short r;
			float pose_theta;
			unsigned short numGoodSpots;
			unsigned short spot0_mag;
			unsigned short spot1_mag;
			unsigned int sequenceNumber;
			unsigned int room;
		};

		class SpotPositionReport
		{
		public:
			SpotPositionReport()
			{
				for( int i=0; i<MAX_NUM_SPOTS; ++i )
				{
					spot_x[i] = 0;
					spot_y[i] = 0;
					spot_mag[i] = 0;
				}
			}

			signed short spot_x[MAX_NUM_SPOTS];
			signed short spot_y[MAX_NUM_SPOTS];
			unsigned short spot_mag[MAX_NUM_SPOTS];
		};

		class MagnitudeReport
		{
		public:
			MagnitudeReport()
			{
				for( int i=0; i<MAX_NUM_SPOTS; ++i )
				{
					magnitude[i] = 0;
				}
			}

			unsigned short magnitude[MAX_NUM_SPOTS];
		};
	}
}

#endif //_REC_NSTAR_COM_

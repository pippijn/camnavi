//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_COM_H_
#define _REC_ROBOTINO_COM_COM_H_

#include "rec/robotino/com/ComId.h"
#include "rec/robotino/com/RobotinoException.h"
#include "rec/iocontrol/remotestate/SetState.h"
#include "rec/iocontrol/remotestate/SensorState.h"
#include <string>

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class ComImpl;

			/**
			* @brief	Represents a communication device.
			*/
			class
#ifdef WIN32
#  ifdef rec_robotino_com_EXPORTS
		__declspec(dllexport)
#endif
#  ifdef rec_robotino_com2_EXPORTS
		__declspec(dllexport)
#endif
#  ifdef rec_robotino_com3_EXPORTS
		__declspec(dllexport)
#endif
#endif
			Com
			{
			public:
				/** \brief Communication errors.

				\li \c NoError											no error
				\li \c ErrorConnectionRefused				if the connection was refused
				\li \c ErrorSocketSend							if a send to the socket failed
				\li \c ErrorSocketRead							if a read from the socket failed
				\li \c ErrorImageServerCreate				if the creation of the image server failed
				\li \c ErrorImageServerBind					if the image server could not bind to the imagePort()
				\li \c ErrorTimeout									if Robotino did not send something for more than timeout() milliseconds
				\li \c ErrorConnectionBusy					if Robotino is already connected to someone else
				\li \c ErrorClientCreate						if the creation of the client socket fails
				\li \c ErrorHostNotFound						if a DNS lookup fails
				\li \c ErrorUndefined								undefined error
				\li \c ErrorWinsock2Initialization	only on Win32 systems if winsock2 could not be initialised
				*/
				typedef enum
				{
					NoError,
					ErrorConnectionRefused,
					ErrorSocketSend,
					ErrorSocketRead,
					ErrorImageServerCreate,
					ErrorImageServerBind,
					ErrorTimeout,
					ErrorConnectionBusy,
					ErrorClientCreate,
					ErrorHostNotFound,
					ErrorUndefined,
					ErrorWinsock2Initialization,
					ErrorAlreadyConnected
				} Error;

				/** \brief State of connection.

				\li \c NotConnected										No connection to Robotino established
				\li \c Connecting											Connection setup in progress
				\li \c Connected											Connection successfully established
				*/
				typedef enum
				{
					NotConnected = 0,
					HostLookupState = 1,
					Connecting = 2,
					Connected = 3,
					ClosingState = 6
				} ConnectionState;

				/**
				@param useQueuedCallback If set to false, event functions are called directly from the internal thread of this Com object.
				Especially in GUI application it is usefull to set useQueuedCallback to true to ensure that event functions are called when
				calling processEvents().
				@see processEvents, errorEvent, connectedEvent, connectionClosedEvent, connectionStateChangedEvent, updateEvent, modeChangedEvent,
				     Camera::imageReceivedEvent, Info::infoReceivedEvent
				*/
				Com( bool useQueuedCallback = false );
				
				virtual ~Com();

				/**
				* The unique identifier of this communication object.
				*
				* @return	The identifier.
				* @throws	nothing.
				*/
				ComId id() const;

				/**
				* Connects to the server. This function blocks until the connection is established or an error occurs.
				* @param isBlocking If true, this function blocks until the connection is established or an error occurs.
				*										Otherwise the function is non blocking and you have to watch for error or connected callbacks.
				* @throws In blocking mode a ComException is thrown if the client couldn't connect. In non blocking mode throws nothing.
				*/
				void connect( bool isBlocking = true );

				/**
				* Disconnects from the server and disables autoupdating.
				*
				* @throws	nothing.
				*/
				void disconnect();

				/**
				* Test wether the client is connected to the server.
				* @return	TRUE if connected, FALSE otherwise.
				* @throws	nothing.
				*/
				bool isConnected() const;

				/**
				* Sets the address of the server.
				* @param address	The address of the server e.g. "172.26.1.1" or "127.0.0.1"
				                    To connect to Robotino Sim you also have to specify the server port.
									The first simulated Robotino listens at port 8080.
									The address is then "127.0.0.1:8080".
									Without port specification port 80 (Robotino's default port) is used.
				* @see				address
				* @throws			nothing.
				*/
				void setAddress( const char* address );

				/**
				* Returns the currently active server address.
				* 
				* @return	Address set with setAddress
				* @see		setAddress
				* @throws	nothing.
				*/
				const char* address() const;

				/**
				* Sets the destination port of the image streamer. Does nothing if
				* we're already connected to a server.
				* @param port	The destination port of the image streamer e.g. 8080
				* @throws		nothing.
				*/
				void setImageServerPort( unsigned int port = 8080 );

				/**
				Retrieve the port number the local UDP server receiving images is connected to.
				This might be different to the port number set by setImageServerPort if the port
				is not free. This might happen because you are running more than one instance of Com
				(in order to drive more than one Robotino) or after destroying a Com object and then
				creating a new Com object giving the OS not enough time to free the port.
				* @return Image server port number
				* @see setImageServerPort
				* @throws nothing
				*/
				unsigned int imageServerPort() const;

				/**
				* @return The current state of connection
				*/
				ConnectionState connectionState() const;

				/**
				* @return Returns true, if this is a connection in passive mode. Passive mode means that you can read Robotino's sensors.
				          But you do not have access to Robotino's actors.
				* @see modeChangedEvent()
				*/
				bool isPassiveMode() const;

				/**
				Wait until new sensor readings are available.
				@param timeout The time in milliseconds after which this operation should timeout. If timeout is 0, this does
				never timeout.
				@return Returns true if new sensor readings became available while calling this function. Returns false if the operation
				lasts for more than timeout milliseconds. Also returns false if the connection is closed (either by an error or by calling disconnect).
				*/
				bool waitForUpdate( unsigned int timeout = 200 );

				/**
				Set the minimum cycle time of the internal thread communicating with Robotino.
				@param msecs The minimum cycle time in milliseconds. A smaler value will lead to a faster update
							 of sensor readings, but will also increase the CPU load. A larger value will lead to slower
							 update of sensor readings but also saves CPU time. The default value is 30. This is a good value
							 when running programms directly on Robotino. If you have a fast CPU you might set this value to 0 to
							 get the fastest communication possible. Values larger than 100 will be interpreted as 100.
				*/
				void setMinimumUpdateCycleTime( unsigned int msecs );

				rec::iocontrol::remotestate::SensorState sensorState();

				void setSetState( const rec::iocontrol::remotestate::SetState& setState );

				/**
				Call this function from your GUI thread when the Com object was constructed with useQueuedCallback set to true.
				All callback functions are called from here if callback events are pending.
				If useQueuedCallback was false when constructing the Com object this function has no effect.
				@throws nothing
				*/
				void processEvents();

				/**
				Check if there are pending callback events. This function will alwasy return false if the Com object was constructed with useQueuedCallback set to false.
				@return Returns true if there are pending events, i.e. a call to processEvents will result in calling at least one
				callback function.
				@throws nothing
				@see processEvents
				*/
				bool hasPendingEvents() const;

				/**
				* This function is called on errors.
				* Note: This function is called from outside the applications main thread.
				* This is extremely important particularly with regard to GUI applications.
				* @param error The error which caused this event.
				* @param errorString A human readable error description.
				* @throws		nothing.
				*/
				virtual void errorEvent( Error error, const char* errorString );

				/**
				* This function is called if a connection to Robotino has been established.
				* Note: This function is called from outside the applications main thread.
				* This is extremely important particularly with regard to GUI applications.
				* @throws		nothing.
				*/
				virtual void connectedEvent();

				/**
				* This function is called if a connection is closed.
				* Note: This function is called from outside the applications main thread.
				* This is extremely important particularly with regard to GUI applications.
				* @throws		nothing.
				*/
				virtual void connectionClosedEvent();

				/**
				* This function is called whenever the connection state changes
				* @param newState The new connection state
				* @param oldState The previous connection state
				*/
				virtual void connectionStateChangedEvent( ConnectionState newState, ConnectionState oldState );

				/**
				* This function is called whenever sensor readings have been updated
				*/
				virtual void updateEvent();

				/**
				* This function is called whenever the mode of connection changes from passive to active or vice versa.
				* @param isPassiveMode True if the mode changed to passive mode. False otherwise.
				*/
				virtual void modeChangedEvent( bool isPassiveMode );

			private:
				ComImpl* _impl;
			};

			/**
			* @brief	RobotinoException exclusive to Com objects.
			*/
			class ComException : public RobotinoException
			{
			public:
				ComException( Com::Error e, const char* message )
					: RobotinoException( message )
					, error( e )
				{
				}

				const Com::Error error;
			};
		}
	}
}
#endif

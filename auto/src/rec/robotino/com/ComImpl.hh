//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_COMIMPL_H_
#define _REC_ROBOTINO_COM_COMIMPL_H_

#include <QObject>
#include <QMutex>
#include <QMap>
#include <QList>
#include <QAbstractSocket>
#include <QString>
#include <QThread>
#include <QWaitCondition>
#include <QSemaphore>
#include <QHostAddress>
#include <QStringList>
#include <QCoreApplication>

#include <QSharedMemory>
#include "rec/robotino/imagesender/ImageMemory.h"

#include "rec/iocontrol/remotestate/SetState.h"
#include "rec/iocontrol/remotestate/SensorState.h"

#include "rec/robotino/com/RobotinoException.h"
#include "rec/robotino/com/Com.h"

#include "rec/robotino/com/events/ComEvent.h"

#include "rec/core_lt/image/Image.h"
#include "rec/core_lt/image/conv.h"

using rec::iocontrol::remotestate::SetState;
using rec::iocontrol::remotestate::SensorState;

namespace rec
{
	namespace robotino
	{     
		namespace com
		{
			class Com;
			class Camera;
			class JPGCamera;
			class Info;
			class Gripper;
			class ComChild;

			class ComEvent;

			class ComImpl : public QThread
			{
				Q_OBJECT
			public:
				const ComId comid;

				typedef enum { UNKNOWN_TYPE, JPEG, RAW, BMP } ImageType_t;

				/**
				* Retrieves the ComImpl instance corresponding to the given comID
				* @return	The instance. Never returns NULL.
				* @throws	RobotinoException if there is no instance with the given comID
				*/
				static ComImpl* instance( const ComId& id );

				ComImpl( Com* com, bool useQueuedCallback );
				~ComImpl();

				bool update( unsigned int timeout );

				bool waitForUpdate( unsigned int timeout );

				/**
				* @param isBlocking If true, this function blocks until the connection is established or an error occurs.
				*/
				void connect( bool isBlocking );

				void disconnect();

				void setAddress( const char* address );

				const char* address() const;

				void setImageServerPort( unsigned int port );

				unsigned int imageServerPort() const;

				bool isConnected() const;

				Com::ConnectionState connectionState() const;

				void setAutoUpdateEnabled( bool enable );

				bool isAutoUpdateEnabled() const;

				//Camera **************
				void registerStreamingCamera( Camera* camera );
				void deregisterStreamingCamera( Camera* camera );
				bool isStreaming( Camera* camera ) const;
				void registerStreamingCamera( JPGCamera* camera );
				void deregisterStreamingCamera( JPGCamera* camera );
				bool isStreaming( JPGCamera* camera ) const;
			
				volatile bool _cameraResolutionChanged;
				//Camera **************

				//Gripper *************
				void registerGripper( Gripper* gripper );
				void deregisterGripper( Gripper* gripper );
				//Gripper *************

				//Info ****************
				void registerInfo( Info* info );
				void deregisterInfo( Info* info );
				std::string _infoText;
				QMutex _infoTextMutex;
				//Info ****************

				//Com children *************************************
				void registerComChild( ComChild* child );
				void deregisterComChild( ComChild* child );
				//Com children *************************************

				QMutex _setStateMutex;
				SetState _setState;

				QMutex _sensorStateMutex;
				SensorState _sensorState;

				mutable QMutex _workerMutex;

				void processEvents();

				bool hasPendingEvents() const;

				int _msecsPerUpdateCycle;

				bool resetPosition[4];

			private:
				void run();

				void resetSetState();
				void resetSensorState();

				bool isStreaming_i( Camera* camera ) const;
				bool isStreaming_i( JPGCamera* camera ) const;
				int numStreamingCameras_i() const;

				void attachToSharedMemory();
				void detachFromSharedMemory();

				void imageReceived( const QByteArray& imageData,
					int type,
					unsigned int width,
					unsigned int height,
					unsigned int numChannels,
					unsigned int bitsPerChannel,
					unsigned int step );

				void infoReceived( const char* text );

				const bool _useQueuedCallback;

				volatile bool _run;
				volatile bool _keepConnected;

				typedef enum { IdleState, WorkingState } State;

				volatile State _state;

				QWaitCondition _connectCondition;

				QWaitCondition _connectedCondition;

				QWaitCondition _disconnectedCondition;

				QWaitCondition _updateCondition;

				rec::robotino::com::Com* _com;

				std::string _addressString;
				QHostAddress _address;
				quint16 _port;

				static QMap< ComId, ComImpl* > _instances;
				static QMutex _instancesMutex;

				QList< Camera* > _streamingCameras;
				QList< JPGCamera* > _streamingJPGCameras;
				mutable QMutex _streamingCamerasMutex;

				QList< Info* > _infos;
				mutable QMutex _infosMutex;

				QList< Gripper* > _grippers;
				mutable QMutex _grippersMutex;

				QList< ComChild* > _comChildren;
				mutable QMutex _comChildrenMutex;

				Com::ConnectionState _prevState;

				unsigned int _imageServerPort;

				QAbstractSocket::SocketState _socketState;

				QMutex _sharedMemoryMutex;
				QSharedMemory _sharedMemory;
				rec::robotino::imagesender::ImageMemory* _imageMemory;

				mutable QMutex _eventMutex;
				QMap< ComEvent::Id_t, ComEvent* > _events;

				rec::core_lt::image::Image _currentImage;

			Q_SIGNALS:
				void sharedMemoryImageReceived( const QByteArray& imageData,
																		  int type,
																			unsigned int width,
																			unsigned int height,
																			unsigned int numChannels,
																			unsigned int bitsPerChannel,
																			unsigned int step );

			private Q_SLOTS:
				void on_v1Com_stateChanged( QAbstractSocket::SocketState );
				void on_v1Com_infoReceived( const QString& text, bool isPassiveMode );
				void on_v1Com_imageReceived( const QByteArray& imageData,
																		  int type,
																			unsigned int width,
																			unsigned int height,
																			unsigned int numChannels,
																			unsigned int bitsPerChannel,
																			unsigned int step );

				void on_v1Com_error( QAbstractSocket::SocketError socketError );
			};

			void initQt();
		}
	}
}

#endif

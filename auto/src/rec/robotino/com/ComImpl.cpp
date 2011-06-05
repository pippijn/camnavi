//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/ComImpl.hh"
#include "rec/robotino/com/Camera.h"
#include "rec/robotino/com/JPGCamera.h"
#include "rec/robotino/com/Info.h"
#include "rec/robotino/com/Gripper.h"
#include "rec/robotino/com/ComChild.hh"

#include "rec/robotino/com/v1/Com.h"

#include "rec/robotino/com/events/ConnectedEvent.h"
#include "rec/robotino/com/events/ConnectionClosedEvent.h"
#include "rec/robotino/com/events/ConnectionStateChangedEvent.h"
#include "rec/robotino/com/events/ErrorEvent.h"
#include "rec/robotino/com/events/ImageReceivedEvent.h"
#include "rec/robotino/com/events/InfoReceivedEvent.h"
#include "rec/robotino/com/events/ModeChangedEvent.h"
#include "rec/robotino/com/events/UpdateEvent.h"

#include <QMetaType>
#include <QCoreApplication>
#include <QTime>

#include <cassert>

Q_DECLARE_METATYPE( QAbstractSocket::SocketState )

using rec::robotino::com::Com;
using rec::robotino::com::ComException;
using rec::robotino::com::Camera;
using rec::robotino::com::JPGCamera;
using rec::robotino::com::Info;
using rec::robotino::com::ComImpl;
using rec::robotino::com::ComId;
using rec::robotino::com::RobotinoException;
using rec::robotino::com::Gripper;
using rec::robotino::com::ComChild;

QMap<ComId, ComImpl* > ComImpl::_instances;
QMutex ComImpl::_instancesMutex;

ComImpl* ComImpl::instance( const ComId& comID )
{
	QMutexLocker lk( &_instancesMutex );

	ComImpl* p = _instances.value( comID, NULL );

	if( NULL == p )
	{
		throw RobotinoException( "Invalid comID" );
	}
	else
	{
		return p;
	}
}

ComImpl::ComImpl( Com* com, bool useQueuedCallback )
: comid( ComId::g_id++ )
, _useQueuedCallback( useQueuedCallback )
, _run( true )
, _keepConnected( true )
, _com( com )
, _prevState( Com::NotConnected )
, _imageServerPort( 0 )
, _socketState( QAbstractSocket::UnconnectedState )
, _state( IdleState )
, _port( 0 )
, _imageMemory( NULL )
, _cameraResolutionChanged( true )
, _msecsPerUpdateCycle( 30 )
{
	initQt();
	
	qRegisterMetaType<QAbstractSocket::SocketState>();

	QMutexLocker lk( &_instancesMutex );
	_instances[ comid ] = this;

	resetSetState();
	resetSensorState();
}

ComImpl::~ComImpl()
{
	QMutexLocker mlk( &_comChildrenMutex );
	Q_FOREACH( ComChild* m, _comChildren )
	{
		m->disconnectFromServer();
	}

	_workerMutex.lock();
	_run = false;
	_keepConnected = false;
	_connectCondition.wakeAll();
	_connectedCondition.wakeAll();
	_disconnectedCondition.wakeAll();
	_updateCondition.wakeAll();
	_workerMutex.unlock();

	wait();

	QMutexLocker lk( &_instancesMutex );
	_instances.remove( comid );
}

void ComImpl::run()
{
	rec::robotino::com::v1::Com v1Com( NULL );

	QObject::connect( &v1Com, SIGNAL( stateChanged( QAbstractSocket::SocketState ) ), SLOT( on_v1Com_stateChanged( QAbstractSocket::SocketState ) ), Qt::DirectConnection );
	QObject::connect( &v1Com, SIGNAL( infoReceived( const QString&, bool ) ), SLOT( on_v1Com_infoReceived( const QString&, bool ) ), Qt::DirectConnection );
	QObject::connect( &v1Com,
		SIGNAL( imageReceived( const QByteArray&, int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int ) ),
		SLOT( on_v1Com_imageReceived( const QByteArray&, int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int ) ), Qt::DirectConnection );
	QObject::connect( this,
		SIGNAL( sharedMemoryImageReceived( const QByteArray&, int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int ) ),
		SLOT( on_v1Com_imageReceived( const QByteArray&, int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int ) ), Qt::DirectConnection );

	while( _run )
	{
		{
			QMutexLocker lk( &_grippersMutex );
			QMutexLocker lk2( &_streamingCamerasMutex );
			QMutexLocker lk3( &_setStateMutex );
			_imageServerPort = v1Com.imageServerPort();
			_setState.imageServerPort = v1Com.imageServerPort();
			_setState.isImageRequest = ( numStreamingCameras_i() > 0 );
			_setState.gripper_isEnabled = ( false == _grippers.isEmpty() );
			_cameraResolutionChanged = true;
		}

		bool isPassiveMode = false;
		_keepConnected = true;

		unsigned int imageSequenceNumber = 0;

		v1Com.connectToHost( _address, _port );

		if( false == v1Com.waitForConnected( 2000 ) )
		{
			_com->errorEvent( v1Com.comError(), v1Com.errorString().toLatin1().data() );
			QMutexLocker lk( &_workerMutex );
			_connectedCondition.wakeAll();
		}
		else
		{
			{
				QMutexLocker lk( &_workerMutex );
				_state = WorkingState;
				_connectedCondition.wakeAll();
			}

			{
				if( QHostAddress::LocalHost == _address )
				{
					attachToSharedMemory();
				}
			}

			{
				QMutexLocker lk( &_setStateMutex );
				v1Com.sendIOControl( &_setState );
			}

			{
				QMutexLocker lk( &_sensorStateMutex );
				v1Com.update( &_sensorState );
			}

			QTime updateTimer;

			while( _keepConnected )
			{
				updateTimer.start();

				{
					QMutexLocker lk( &_setStateMutex );
					
					if( _cameraResolutionChanged )
					{
						_cameraResolutionChanged = false;
						v1Com.sendCameraControl( _setState.camera_imageWidth, _setState.camera_imageHeight );
					}

					for( int i=0; i<3; ++i )
					{
						if( resetPosition[i] )
						{
							_setState.resetPosition[i] = true;
							resetPosition[i] = false;
						}
					}

					if( resetPosition[3] )
					{
						_setState.encoderInputResetPosition = true;
						resetPosition[3] = false;
					}

					v1Com.sendIOControl( &_setState );

					for( int i=0; i<3; ++i )
					{
						if( resetPosition[i] )
						{
							_setState.resetPosition[i] = false;
						}
					}

					_setState.encoderInputResetPosition = false;
				}

				bool ret;

				{
					QMutexLocker lk( &_sensorStateMutex );
					ret = v1Com.update( &_sensorState );
				}

				if( ret )
				{
					_updateCondition.wakeAll();

					{
						QMutexLocker lk( &_eventMutex );
						UpdateEvent* e = new UpdateEvent;
						_events[e->id] = e;
					}

					if( _sensorState.isPassiveMode != isPassiveMode )
					{
						isPassiveMode = _sensorState.isPassiveMode;

						{
							QMutexLocker lk( &_eventMutex );
							ModeChangedEvent* e = new ModeChangedEvent( isPassiveMode );
							_events[e->id] = e;
						}
					}

					{
						QMutexLocker lock( &_sharedMemoryMutex );
						if( NULL != _imageMemory )
						{
							if( _sharedMemory.lock() )
							{
								if( imageSequenceNumber != _imageMemory->sequenceNumber )
								{
									imageSequenceNumber = _imageMemory->sequenceNumber;

									QByteArray ba( (const char*)_imageMemory->data, _imageMemory->dataSize );
									Q_EMIT sharedMemoryImageReceived( ba, RAW, _imageMemory->width, _imageMemory->height, _imageMemory->numChannels, _imageMemory->bitsPerChannel, _imageMemory->step );
								}
								_sharedMemory.unlock();
							}
							else
							{
								qDebug() << "Error locking shared memory";
							}
						}
					}
				}
				else
				{
					_com->errorEvent( rec::robotino::com::Com::ErrorTimeout, "Timeout reading from socket." );
					break;
				}

				if( false == _useQueuedCallback )
				{
					processEvents();
				}

				if( _keepConnected )
				{
					int msecsElapsed = updateTimer.elapsed();
					//qDebug() << msecsElapsed;
					if( msecsElapsed < _msecsPerUpdateCycle )
					{
						int diff = _msecsPerUpdateCycle - msecsElapsed;
						//qDebug() << msecsElapsed << "  " << diff;
						msleep( diff );
					}
				}
			}

			resetSetState();
			resetSensorState();
		}

		detachFromSharedMemory();

		v1Com.disconnectFromHost();

		if( false == _useQueuedCallback )
		{
			processEvents();
		}

		{
			QMutexLocker lk( &_workerMutex );
			_state = IdleState;
			_disconnectedCondition.wakeAll();
			_updateCondition.wakeAll();
			if( false == _run )
			{
				return;
			}
			_connectCondition.wait( &_workerMutex );
		}
	}
}

bool ComImpl::waitForUpdate( unsigned int timeout )
{
	QMutexLocker lk( &_workerMutex );

	if( IdleState == _state )
	{
		return false;
	}

	unsigned long t = timeout;
	if( 0 == t )
	{
		t = ULONG_MAX;
	}

	bool ret = _updateCondition.wait( &_workerMutex, t );

	if( false == ret || IdleState == _state )
	{
		return false;
	}
	else
	{
		return true;
	}
}

void ComImpl::connect( bool isBlocking )
{
	QMutexLocker lk( &_workerMutex );

	if( false == _run )
	{
		if( isBlocking )
		{
			throw rec::robotino::com::ComException( rec::robotino::com::Com::ErrorUndefined, "Connecting while destroying the Com object." );
		}
	}
	
	if( !isRunning() )
	{
		start( QThread::TimeCriticalPriority );
	}
	else
	{
		if( WorkingState == _state )
		{
			_com->errorEvent( rec::robotino::com::Com::ErrorAlreadyConnected, "Connect called while connecting is already established." );
			if( isBlocking )
			{
				throw rec::robotino::com::ComException( rec::robotino::com::Com::ErrorAlreadyConnected, "Connect called while connecting is already established." );
			}
		}
		else if( IdleState == _state )
		{
			_connectCondition.wakeOne();
		}
		else
		{
			assert( false );
		}
	}

	if( isBlocking )
	{
		_connectedCondition.wait( &_workerMutex );

		if( IdleState == _state )
		{
			if( isBlocking )
			{
				throw rec::robotino::com::ComException( rec::robotino::com::Com::ErrorConnectionRefused, "The connection has been refused." );
			}
		}
	}

	QMutexLocker mlk( &_comChildrenMutex );
	Q_FOREACH( ComChild* m, _comChildren )
	{
		m->connectToServer();
	}
}

void ComImpl::disconnect()
{
	QMutexLocker lk( &_workerMutex );

	if( false == _run )
	{
		return;
	}

	if( WorkingState == _state )
	{
		_keepConnected = false;
		_disconnectedCondition.wait( &_workerMutex );
	}

	QMutexLocker mlk( &_comChildrenMutex );
	Q_FOREACH( ComChild* m, _comChildren )
	{
		m->disconnectFromServer();
	}
}

void ComImpl::setAddress( const char* address )
{
	_addressString = address;

	QString str( address );

	QStringList l = str.split( ':' );

	_address = QHostAddress( l.first() );

	if( l.size() > 1 )
	{
		_port = l.at(1).toUInt();
	}
	else
	{
		_port = 80;
	}

	QMutexLocker lk( &_comChildrenMutex );
	Q_FOREACH( ComChild* m, _comChildren )
	{
		m->setAddress( address );
	}
}

const char* ComImpl::address() const
{
	return _addressString.c_str();
}

void ComImpl::setImageServerPort( unsigned int port )
{
}

unsigned int ComImpl::imageServerPort() const
{
	return _imageServerPort;
}

bool ComImpl::isConnected() const
{
	return ( WorkingState == _state );
}

Com::ConnectionState ComImpl::connectionState() const
{
	return static_cast< Com::ConnectionState >( _socketState );
}

void ComImpl::resetSetState()
{
	for( int i=0; i<4; ++i )
	{
		resetPosition[i] = false;
	}

	QMutexLocker lk( &_setStateMutex );
	_setState.reset();
}

void ComImpl::resetSensorState()
{
	QMutexLocker lk( &_sensorStateMutex );
	_sensorState.reset();
}

void ComImpl::registerInfo( Info* info )
{
	QMutexLocker lk( &_infosMutex );
	_infos.append( info );
}

void ComImpl::deregisterInfo( Info* info )
{
	QMutexLocker lk( &_infosMutex );
	_infos.removeAll( info );
}

void ComImpl::registerComChild( ComChild* child )
{
	QMutexLocker lk( &_comChildrenMutex );
	_comChildren.append( child );

	child->setAddress( _addressString.c_str() );

	if( isConnected() )
	{
		child->connectToServer();
	}
}

void ComImpl::deregisterComChild( ComChild* child )
{
	QMutexLocker lk( &_comChildrenMutex );
	_comChildren.removeAll( child );

	child->disconnectFromServer();
}

void ComImpl::registerStreamingCamera( Camera* camera )
{
	QMutexLocker lock( &_streamingCamerasMutex );

	if( false == isStreaming_i( camera ) )
	{
		_streamingCameras.append( camera );
	}
	else
	{
		return;
	}

	{
		QMutexLocker lk2( &_setStateMutex );
		_setState.isImageRequest = ( numStreamingCameras_i() > 0 );
	}

	{
		QMutexLocker lk2( &_sharedMemoryMutex );
		if( _imageMemory )
		{
			_imageMemory->registerRawRequest();
		}
	}
}

void ComImpl::deregisterStreamingCamera( Camera* camera )
{
	QMutexLocker lock( &_streamingCamerasMutex );
	_streamingCameras.removeAll( camera );

	{
		QMutexLocker lk2( &_setStateMutex );
		_setState.isImageRequest = ( numStreamingCameras_i() > 0 );
	}

	{
		QMutexLocker lk2( &_sharedMemoryMutex );
		if( _imageMemory )
		{
			_imageMemory->unregisterRawRequest();
		}
	}
}

bool ComImpl::isStreaming( Camera* camera ) const
{
	QMutexLocker lk( &_streamingCamerasMutex );
	return isStreaming_i( camera );
}

bool ComImpl::isStreaming_i( Camera* camera ) const
{
	return _streamingCameras.contains( camera );
}

void ComImpl::registerStreamingCamera( JPGCamera* camera )
{
	QMutexLocker lock( &_streamingCamerasMutex );

	if( false == isStreaming_i( camera ) )
	{
		_streamingJPGCameras.append( camera );
	}
	else
	{
		return;
	}

	{
		QMutexLocker lk2( &_setStateMutex );
		_setState.isImageRequest = ( numStreamingCameras_i() > 0 );
	}

	{
		QMutexLocker lk2( &_sharedMemoryMutex );
		if( _imageMemory )
		{
			_imageMemory->registerJpgRequest();
		}
	}
}

void ComImpl::deregisterStreamingCamera( JPGCamera* camera )
{
	QMutexLocker lock( &_streamingCamerasMutex );
	_streamingJPGCameras.removeAll( camera );

	{
		QMutexLocker lk2( &_setStateMutex );
		_setState.isImageRequest = ( numStreamingCameras_i() > 0 );
	}

	{
		QMutexLocker lk2( &_sharedMemoryMutex );
		if( _imageMemory )
		{
			_imageMemory->unregisterJpgRequest();
		}
	}
}

bool ComImpl::isStreaming( JPGCamera* camera ) const
{
	QMutexLocker lk( &_streamingCamerasMutex );
	return isStreaming_i( camera );
}

bool ComImpl::isStreaming_i( JPGCamera* camera ) const
{
	return _streamingJPGCameras.contains( camera );
}

int ComImpl::numStreamingCameras_i() const
{
	return _streamingCameras.size() + _streamingJPGCameras.size();
}

void ComImpl::attachToSharedMemory()
{
	QMutexLocker lk3( &_streamingCamerasMutex );
	QMutexLocker lk( &_setStateMutex );
	QMutexLocker lk2( &_sharedMemoryMutex );

	if( _sharedMemory.isAttached() )
	{
		qDebug() << "Shared memoy should not be attached when calling attachToSharedMemory";
		return;
	}

	_sharedMemory.setKey( REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_KEY( _port ) );
	if( _sharedMemory.attach() )
	{
		qDebug() << "Successfully attached to image memory";
		_imageMemory = static_cast<rec::robotino::imagesender::ImageMemory*>( _sharedMemory.data() );
		_setState.imageServerPort = 0;
		_imageServerPort = 0;

		qDebug() << _imageMemory->width;
		qDebug() << _imageMemory->height;

		_imageMemory->registerRawRequest( _streamingCameras.size() );
		_imageMemory->registerJpgRequest( _streamingJPGCameras.size() );
	}
}

void ComImpl::detachFromSharedMemory()
{
	QMutexLocker lk3( &_streamingCamerasMutex );
	QMutexLocker lk2( &_sharedMemoryMutex );

	if( _sharedMemory.isAttached() )
	{
		_imageMemory->unregisterRawRequest( _streamingCameras.size() );
		_imageMemory->unregisterJpgRequest( _streamingJPGCameras.size() );

		_sharedMemory.detach();
		_imageMemory = NULL;
	}
}

void ComImpl::registerGripper( Gripper* gripper )
{
	QMutexLocker lk( &_grippersMutex );
	_grippers.append( gripper );

	QMutexLocker lk2( &_setStateMutex );
	_setState.gripper_isEnabled = true;
}
				
void ComImpl::deregisterGripper( Gripper* gripper )
{
	QMutexLocker lk( &_grippersMutex );
	_grippers.removeAll( gripper );

	QMutexLocker lk2( &_setStateMutex );
	_setState.gripper_isEnabled = ( false == _grippers.isEmpty() );
}

void ComImpl::on_v1Com_stateChanged( QAbstractSocket::SocketState state )
{
	_socketState = state;

	_com->connectionStateChangedEvent( static_cast<Com::ConnectionState>( state ), _prevState );
	_prevState = static_cast<Com::ConnectionState>( state );

	switch( state )
	{
	case QAbstractSocket::ConnectedState:
		{
			ConnectedEvent *e = new ConnectedEvent;
			QMutexLocker lk( &_eventMutex );
			_events[e->id] = e;
		}
		break;

	case QAbstractSocket::UnconnectedState:
		{
			ConnectionClosedEvent *e = new ConnectionClosedEvent;
			QMutexLocker lk( &_eventMutex );
			_events[e->id] = e;
		}
		break;

	default:
		break;
	}
}

void ComImpl::infoReceived( const char* text )
{
	QMutexLocker lk( &_infosMutex );
	Q_FOREACH( Info* info, _infos )
	{
		info->infoReceivedEvent( text );
	}
}

void ComImpl::on_v1Com_infoReceived( const QString& text, bool isPassiveMode )
{
	{
		QMutexLocker lk( &_infoTextMutex );
		_infoText = text.toLatin1().constData();
	}

	{
		InfoReceivedEvent *e = new InfoReceivedEvent;
		QMutexLocker lk( &_eventMutex );
		_events[e->id] = e;
	}
}

void ComImpl::imageReceived( const QByteArray& imageData,
																		  int type,
																			unsigned int width,
																			unsigned int height,
																			unsigned int numChannels,
																			unsigned int bitsPerChannel,
																			unsigned int step )
{
	QMutexLocker lock( &_streamingCamerasMutex );

	if( RAW == (ImageType_t)type )
	{
		Q_FOREACH( Camera* camera, _streamingCameras )
		{
			camera->imageReceivedEvent( reinterpret_cast<const unsigned char*>( imageData.constData() ),
				imageData.size(),
				width,
				height,
				numChannels,
				bitsPerChannel,
				step );
		}
	}
	else
	{
		Q_FOREACH( JPGCamera* camera, _streamingJPGCameras )
		{
			camera->jpgReceivedEvent( reinterpret_cast<const unsigned char*>( imageData.constData() ),
				imageData.size() );
		}

		if( false == _streamingCameras.isEmpty() )
		{
			rec::core_lt::memory::ByteArrayConst ba = rec::core_lt::memory::ByteArrayConst::fromRawData( reinterpret_cast<const unsigned char*>( imageData.constData() ), imageData.size() );
			_currentImage = rec::core_lt::image::loadFromData( ba, "jpg" );

			if( false == _currentImage.isNull() )
			{
				Q_FOREACH( Camera* camera, _streamingCameras )
				{
					camera->imageReceivedEvent( _currentImage.constData(),
						_currentImage.step() * _currentImage.info().height,
						_currentImage.info().width,
						_currentImage.info().height,
						_currentImage.info().numChannels,
						_currentImage.info().bytesPerChannel * 8,
						_currentImage.step() );
				}
			}
		}
	}
}

void ComImpl::on_v1Com_imageReceived( const QByteArray& imageData,
																		  int type,
																			unsigned int width,
																			unsigned int height,
																			unsigned int numChannels,
																			unsigned int bitsPerChannel,
																			unsigned int step )
{
	ImageReceivedEvent* e = new ImageReceivedEvent( imageData, (rec::robotino::com::ComImpl::ImageType_t)type, width, height, numChannels, bitsPerChannel, step );
	QMutexLocker lk( &_eventMutex );
	_events[e->id] = e;
}

void ComImpl::on_v1Com_error( QAbstractSocket::SocketError socketError )
{
	//_com->errorEvent( _v1Com->comError(), _v1Com->errorString().toLatin1().data() );
}

void ComImpl::processEvents()
{
	QMutexLocker lk( &_eventMutex );
	QMap< ComEvent::Id_t, ComEvent* >::const_iterator iter = _events.constBegin();
	while( _events.constEnd() != iter )
	{
		switch( iter.key() )
		{
		case ComEvent::ConnectedEventId:
			_com->connectedEvent();
			break;

		case ComEvent::ConnectionClosedEventId:
			_com->connectionClosedEvent();
			break;

		case ComEvent::ConnectionStateChangedEventId:
			{
				ConnectionStateChangedEvent* e = static_cast<ConnectionStateChangedEvent*>( iter.value() );
				_com->connectionStateChangedEvent( e->newState, e->oldState );
			}
			break;

		case ComEvent::ErrorEventId:
			{
				ErrorEvent* e = static_cast<ErrorEvent*>( iter.value() );
				_com->errorEvent( e->error, e->errorStr.c_str() );
			}
			break;

		case ComEvent::ImageReceivedEventId:
			{
				ImageReceivedEvent* e = static_cast<ImageReceivedEvent*>( iter.value() );
				imageReceived( e->data, e->type, e->width, e->height, e->numChannels, e->bitsPerChannel, e->step );
			}
			break;

		case ComEvent::InfoReceivedEventId:
			{
				InfoReceivedEvent* e = static_cast<InfoReceivedEvent*>( iter.value() );
				QMutexLocker lk( &_infoTextMutex );
				infoReceived( _infoText.c_str() );
			}
			break;

		case ComEvent::ModeChangedEventId:
			{
				ModeChangedEvent* e = static_cast<ModeChangedEvent*>( iter.value() );
				_com->modeChangedEvent( e->isPassiveMode );
			}
			break;

		case ComEvent::UpdateEventId:
			{
				_com->updateEvent();
			}
			break;

		default:
			qDebug() << "Unhandled event id " << iter.key();
			break;
		}

		++iter;
	}

	qDeleteAll( _events );
	_events.clear();
}

bool ComImpl::hasPendingEvents() const
{
	QMutexLocker lk( &_eventMutex );
	return _events.size();
}

void rec::robotino::com::initQt()
{
	if( NULL == QCoreApplication::instance() )
	{
		static int argc = 1;
		static char a[2];
		static char b[2];
		static char* argv[2] = { a, b };
		strcpy( a, "a" );
		strcpy( b, "b" );
		new QCoreApplication( argc, argv );
	}
}

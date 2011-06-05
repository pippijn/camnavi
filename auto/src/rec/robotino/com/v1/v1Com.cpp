#include "rec/robotino/com/v1/Com.h"
#include "rec/robotino/com/v1/messages/Info.h"
#include "rec/robotino/com/v1/messages/IOControl.h"
#include "rec/robotino/com/v1/messages/IOStatus.h"
#include "rec/robotino/com/v1/messages/CameraControl.h"

#include "rec/robotino/com/v1/ImageServer.h"

#include <QtDebug>

using namespace rec::robotino::com::v1;

Com::Com( QObject* parent )
: QTcpSocket( parent )
, _imageServer( new ImageServer( this ) )
, _readTimeout( 2000 )
{
	connect( this, SIGNAL( stateChanged( QAbstractSocket::SocketState ) ), SLOT( on_stateChanged( QAbstractSocket::SocketState ) ) );
	connect( this, SIGNAL( bytesWritten( qint64 ) ), SLOT( on_bytesWritten( qint64 ) ) );
	connect( this, SIGNAL( readyRead() ), SLOT( on_readyRead() ) );

	connect( _imageServer,
		SIGNAL( imageReceived( const QByteArray&, int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int ) ),
		SIGNAL( imageReceived( const QByteArray&, int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int ) ) );

	for( unsigned int i=0; i<100; ++i )
	{
		if( _imageServer->bind( 8080+i ) )
		{
			break;
		}
	}
}

void Com::sendIOControl( rec::iocontrol::remotestate::SetState* setState )
{
	_gripper.set( setState );

	QByteArray data = messages::IOControl::encode( *setState );
	write( data );
	waitForBytesWritten();

	setState->setOdometry = false;
}

void Com::sendCameraControl( unsigned int width, unsigned int height )
{
	QByteArray data = messages::CameraControl::encode( width, height );
	write( data );
	waitForBytesWritten();
}

bool Com::update( rec::iocontrol::remotestate::SensorState* sensorState )
{
	while( bytesAvailable() < 3 )
	{
		if( false == waitForReadyRead( _readTimeout ) )
		{
			return false;
		}
	}

	QByteArray headerData = read( 3 );

	quint8 messageId = headerData.at( 0 );
	quint16 messageLength = static_cast<unsigned char>( headerData.at( 1 ) );
	messageLength |= ( static_cast<unsigned char>( headerData.at( 2 ) ) << 8 );

	while( bytesAvailable() < messageLength )
	{
		if( false == waitForReadyRead( _readTimeout ) )
		{
			return false;
		}
	}

	QByteArray messageData = read( messageLength );

	switch( messageId )
	{
	case 1: //IOStatus message
		{
			messages::IOStatus::decode( messageData, sensorState );
			_gripper.set( sensorState );
		}
		break;

	case 6: //Info message
		{
			messages::Info info( messageData );
			Q_EMIT infoReceived( info.text(), info.isPassiveMode() );
		}
		break;

	default:
		break;
	}

	_imageServer->update();

	return true;
}

void Com::on_stateChanged( QAbstractSocket::SocketState socketState )
{
	if( QAbstractSocket::UnconnectedState )
	{
		_gripper.reset();
	}

	qDebug() << socketState;
}

rec::robotino::com::Com::Error Com::comError() const
{
	switch( error() )
	{
	case QAbstractSocket::ConnectionRefusedError:
		return rec::robotino::com::Com::ErrorConnectionRefused;
		break;

	case QAbstractSocket::HostNotFoundError:
		return rec::robotino::com::Com::ErrorHostNotFound;
		break;

	default:
		return rec::robotino::com::Com::ErrorUndefined;
	}
}

void Com::on_bytesWritten( qint64 bytes )
{
	//qDebug() << "bytes written " << bytes;
}

void Com::on_readyRead()
{
	//qDebug() << "readyRead";
}

unsigned int Com::imageServerPort() const
{
	return _imageServer->localPort();
}

#include "rec/robotino/imagesender/Sender.h"
#include <QtDebug>

using namespace rec::robotino::imagesender;

const int Sender::_startSequenceSize = 10;
const qint8 Sender::_startSequence[10] = {'S','t','a','r','t','S','t','a','r','t'};

const int Sender::_stopSequenceSize = 10;
const qint8 Sender::_stopSequence[10] = {'S','t','o','p','p','S','t','o','p','p'};

Sender::Sender( QObject* parent )
: QUdpSocket( parent )
, _bytesWritten( 0 )
, _receiverPort( 0 )
, _partSize( 20000 )
, _state( IdleState )
, _header( 26, 0 )
, _timer( new QTimer( this ) )
{
	memcpy( (void*)_header.data(), _startSequence, _startSequenceSize );

	_timer->setInterval( 20 );
	_timer->setSingleShot( false );
	connect( _timer, SIGNAL( timeout() ), SLOT( on_timer_timeout() ) );
}

void Sender::setRawImageData( const QByteArray& data,
																unsigned int width,
																unsigned int height,
																unsigned int numChannels,
																unsigned int bitsPerChannel,
																unsigned int step )
{
	setImageData( false, data, width, height, numChannels, bitsPerChannel, step );
}

void Sender::setJpgImageData( const QByteArray& data )
{
	setImageData( true, data, 0, 0, 0, 0, 0 );
}

void Sender::setImageData( bool isJpg, const QByteArray& data,
																unsigned int width,
																unsigned int height,
																unsigned int numChannels,
																unsigned int bitsPerChannel,
																unsigned int step )
{
	if( IdleState != _state )
	{
		qDebug() << "calling setImageData while sending image is not permitted";
		return;
	}

	_imageData = data;
	_bytesWritten = 0;

	if( false == _imageData.isEmpty() && false == _receiver.isNull() )
	{
		_state = SendingState;

		quint32 dataSize = static_cast<unsigned int>( data.size() );

		_header[10] = ( dataSize & 0xFF );
		_header[11] = ( ( dataSize >> 8 ) & 0xFF );
		_header[12] = ( ( dataSize >> 16 ) & 0xFF );
		_header[13] = ( ( dataSize >> 24 ) & 0xFF );

		if( isJpg )
		{
			_header[14] = 0; //JPG
		}
		else
		{
			_header[14] = 2; //RAW
		}

		if( 320 == width && 240 == height )
		{
			_header[15] = 0; //QVGA
		}
		else if( 640 == width && 480 == height )
		{
			_header[15] = 1; //VGA
		}
		else
		{
			_header[15] = 2; //Custom
		}

		_header[16] = ( width & 0xFF );
		_header[17] = ( ( width >> 8 ) & 0xFF );
		_header[18] = ( ( width >> 16 ) & 0xFF );
		_header[19] = ( ( width >> 24 ) & 0xFF );

		_header[20] = ( height & 0xFF );
		_header[21] = ( ( height >> 8 ) & 0xFF );
		_header[22] = ( ( height >> 16 ) & 0xFF );
		_header[23] = ( ( height >> 24 ) & 0xFF );

		_header[24] = numChannels;
		_header[25] = bitsPerChannel;

		write( _header );

		_timer->start();
	}
}

void Sender::setReceiver( quint32 address, quint16 port )
{
	_receiver = QHostAddress( address );
	_receiverPort = port;

	connectToHost( _receiver, port );
	waitForConnected();
}

void Sender::stop()
{
	_timer->stop();
	_state = IdleState;
	_imageData.clear();
	//disconnect( this, SIGNAL( bytesWritten( qint64 ) ), this, SLOT( on_bytesWritten( qint64 ) ) );
}

void Sender::on_timer_timeout()
{
	if( _bytesWritten < _imageData.size() )
	{
		qint32 numBytesToWrite = _partSize;
		if( _bytesWritten + numBytesToWrite >= _imageData.size() )
		{
			numBytesToWrite = _imageData.size() - _bytesWritten;
		}

		write( _imageData.constData() + _bytesWritten, numBytesToWrite );

		_bytesWritten += numBytesToWrite;

		//qDebug() << "Writing " << _bytesWritten;

		//if( -1 == writeDatagram( _imageData.constData() + _bytesWritten, numBytesToWrite, _receiver, _receiverPort ) )
		//{
		//	qDebug() << "Error sending image data " << error();
		//	stop();
		//	return;
		//}
	}
	else
	{
		_timer->stop();
		
		write( (const char*)_stopSequence, _stopSequenceSize );

		_state = IdleState;
		/*emit*/ imageSendingCompleted();
	}
}

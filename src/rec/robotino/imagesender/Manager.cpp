#include "rec/robotino/imagesender/Manager.h"
#include <QtDebug>

#ifdef QT_NO_KEYWORDS
#define MY_FOREACH Q_FOREACH
#else
#define MY_FOREACH foreach
#endif

using namespace rec::robotino::imagesender;

Manager::Manager( QObject* parent )
: _numSendesOnSetImageData( 0 )
, _numLocalReceivers( 0 )
, _imageMemory( NULL )
{
}

Manager::~Manager()
{
	reset();
}


Sender::State Manager::state() const
{
	QMap< Receiver, Sender* >::const_iterator iter = _senders.constBegin();
	while( _senders.constEnd() != iter )
	{
		if( Sender::SendingState == (*iter)->state() )
		{
			return Sender::SendingState;
		}
		++iter;
	}

	return Sender::IdleState;
}

void Manager::stop()
{
	QMap< Receiver, Sender* >::iterator iter = _senders.begin();
	while( _senders.end() != iter )
	{
		(*iter)->stop();
		++iter;
	}
}

void Manager::createSharedMemory( quint16 serverPort )
{
	releaseSharedMemory();

	_sharedMemory.setKey( REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_KEY( serverPort ) );

	if( false == _sharedMemory.create( sizeof( ImageMemory ) ) )
	{
		qDebug() << "Unable to create shared memory";
	}
	else
	{
		qDebug() << "Successfully created shared memory";
		_imageMemory = static_cast<ImageMemory*>( _sharedMemory.data() );

		if( _sharedMemory.lock() )
		{
			_imageMemory->reset();
			_sharedMemory.unlock();
		}
		else
		{
			qDebug() << "Error locking shared memory";
		}
	}
}

void Manager::releaseSharedMemory()
{
	if( _sharedMemory.isAttached() )
	{
		_sharedMemory.detach();
		_imageMemory = NULL;
		qDebug() << "Detached from shared memory";
	}
}

void Manager::setRawImageData( const QByteArray& data,
																unsigned int width,
																unsigned int height,
																unsigned int numChannels,
																unsigned int bitsPerChannel,
																unsigned int step )
{
	_numSendesOnSetImageData = _senders.size();

	QMap< Receiver, Sender* >::iterator iter = _senders.begin();
	while( _senders.end() != iter )
	{
		(*iter)->setRawImageData( data, width, height, numChannels, bitsPerChannel, step );
		++iter;
	}
}

void Manager::setJpgImageData( const QByteArray& data )
{
	_numSendesOnSetImageData = _senders.size();

	QMap< Receiver, Sender* >::iterator iter = _senders.begin();
	while( _senders.end() != iter )
	{
		(*iter)->setJpgImageData( data );
		++iter;
	}
}


void Manager::setLocalRawImageData( const QByteArray& data,
																unsigned int width,
																unsigned int height,
																unsigned int numChannels,
																unsigned int bitsPerChannel,
																unsigned int step )
{
	if( NULL == _imageMemory )
	{
		qDebug() << "ImageMemory is NULL";
		return;
	}

	if( _sharedMemory.lock() )
	{
		_imageMemory->width = width;
		_imageMemory->height = height;
		_imageMemory->numChannels = numChannels;
		_imageMemory->bitsPerChannel = bitsPerChannel;
		_imageMemory->step = step;
		_imageMemory->sequenceNumber++;
		_imageMemory->dataSize = data.size();

		memcpy( _imageMemory->data, data.constData(), data.size() );

		_sharedMemory.unlock();
	}
	else
	{
		qDebug() << "Error locking shared memory";
	}
}

void Manager::setLocalJpgImageData( const QByteArray& data )
{
	if( NULL == _imageMemory )
	{
		qDebug() << "ImageMemory is NULL";
		return;
	}

	if( _sharedMemory.lock() )
	{
		_imageMemory->jpgSequenceNumber++;
		_imageMemory->jpgDataSize = data.size();

		memcpy( _imageMemory->jpgData, data.constData(), data.size() );

		_sharedMemory.unlock();
	}
	else
	{
		qDebug() << "Error locking shared memory";
	}
}

void Manager::addReceiver( quint32 address, quint16 port )
{
	Receiver r( address, port );
	if( _senders.contains( r ) )
	{
		MY_FOREACH( const Receiver& r, _senders.keys() )
		{
			qDebug() << r.address << "  " << r.port;
		}

		qDebug() << "Image receiver " << address << " at port " << port << " is already in the list of receivers.";
		return;
	}

	if( port > 0 )
	{
		Sender* s = new Sender( this );
		s->setReceiver( address, port );

		_senders[r] = s;

		connect( s, SIGNAL( imageSendingCompleted() ), SLOT( on_sender_imageSendingCompleted() ) );
	}
	else
	{
		++_numLocalReceivers;
	}
}

void Manager::removeReceiver( quint32 address, quint16 port )
{
	if( port > 0 )
	{
		Receiver r( address, port );
		Sender* s = _senders.take( r );
		delete s;
	}
	else
	{
		if( _numLocalReceivers > 0 )
		{
			--_numLocalReceivers;
		}
		else
		{
			qDebug() << "Unregister local receiver while _numLocalReceivers=0";
		}
	}
}

void Manager::reset()
{
	qDeleteAll( _senders );
	_senders.clear();
}

void Manager::on_sender_imageSendingCompleted()
{
	--_numSendesOnSetImageData;
	if( 0 == _numSendesOnSetImageData ) //all senders reported imageSendingCompleted
	{
		/*emit*/ imageSendingCompleted();
	}
}

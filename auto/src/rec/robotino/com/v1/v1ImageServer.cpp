#include "rec/robotino/com/v1/ImageServer.h"

using namespace rec::robotino::com::v1;

const unsigned int ImageServer::_startSequenceSize = 10;
const quint8 ImageServer::_startSequence[10] = {'S','t','a','r','t','S','t','a','r','t'};

const unsigned int ImageServer::_stopSequenceSize = 10;
const quint8 ImageServer::_stopSequence[10] = {'S','t','o','p','p','S','t','o','p','p'};

const unsigned int ImageServer::_headerSize = 26;

ImageServer::ImageServer( QObject* parent )
: QUdpSocket( parent )
, _imageType( rec::robotino::com::ComImpl::UNKNOWN_TYPE )
, _width( 0 )
, _height( 0 )
, _numColorChannels( 0 )
, _bitsPerChannel( 0 )
, _offset( 0 )
, _nparts( 0 )
, _startOffset( 0 )
{
}

void ImageServer::update()
{
	while( hasPendingDatagrams() )
	{
		QByteArray data;
		data.resize( pendingDatagramSize() );

		readDatagram( data.data(), data.size() );

		processDatagram( data );
	}
}

void ImageServer::processDatagram( const QByteArray& data )
{
  bool start;
  bool stop;

  start = decodeHeader( data );
  stop = findStopSequence( data );

  _startOffset = 0;

  if( start )
  {
    _offset = 0;
    _nparts = 0;
    _startOffset =_headerSize;
  }

  if( stop )
  {
    unsigned int bytesToCopy = data.size() - _stopSequenceSize - _startOffset;
    if( _offset + bytesToCopy == _imageData.size() )
    {
			
			memcpy( (void*)( _imageData.data() + _offset ), (const void*)( data.constData() + _startOffset ), bytesToCopy );

			Q_EMIT imageReceived( _imageData, _imageType, _width, _height, _numColorChannels, _bitsPerChannel, _width*_numColorChannels );
    }
    else
    {
      //error
    }
  }
  if( start && ( stop == false ) )
  {
    int bytesToCopy = data.size() - _startOffset;
    if( bytesToCopy <= _imageData.size() )
    {
			memcpy( (void*)_imageData.data(), (const void*)( data.constData() + _startOffset ), bytesToCopy );
      _offset = bytesToCopy;
    }
    else
    {
      //error
    }
  }
  if( ( start == false ) && ( stop == false ) )
  {
    if( data.size() + _offset <= _imageData.size() )
    {
      _nparts++;
      memcpy( (void*)( _imageData.data() + _offset ), (const void*)data.constData(), data.size() );
      _offset += data.size();
    }
    else
    {
      //error
    }
  }
}

bool ImageServer::decodeHeader( const QByteArray& data )
{
  if( data.size() < _headerSize )
  {
    return false;
  }

  for( unsigned int i=0; i<_startSequenceSize; i++ )
  {
    if( data[i] != _startSequence[i] )
    {
      return false;
    }
  }

	unsigned int imageDataSize = 0;
  
	imageDataSize = static_cast<unsigned char>( data[_startSequenceSize] );
	imageDataSize |= ( static_cast<unsigned char>( data[_startSequenceSize+1] ) << 8 );
	imageDataSize |= ( static_cast<unsigned char>( data[_startSequenceSize+2] ) << 16 );
	imageDataSize |= ( static_cast<unsigned char>( data[_startSequenceSize+3] ) << 24 );

	_imageData.resize( imageDataSize );

	unsigned char imageType = static_cast<unsigned char>( data[_startSequenceSize+4] );

	switch( imageType )
	{
	case 0: //JPG
		_imageType = rec::robotino::com::ComImpl::JPEG;
		break;
		
	//case 1: //JPG2000
		
	case 2: //RAW
		_imageType = rec::robotino::com::ComImpl::RAW;
		break;
		
	case 3: //BMP
		_imageType = rec::robotino::com::ComImpl::BMP;
		break;
		
	// case 4: //PNG
		
	// case 5: //TIFF
		
	default: //UnknownImageType
		_imageType = rec::robotino::com::ComImpl::UNKNOWN_TYPE;
		break;
	}

	unsigned char resolution = static_cast<unsigned char>( data[_startSequenceSize+5] );

	_width = 0;

	_width = static_cast<unsigned char>( data[_startSequenceSize+6] );
	_width |= ( static_cast<unsigned char>( data[_startSequenceSize+7] ) << 8 );
	_width |= ( static_cast<unsigned char>( data[_startSequenceSize+8] ) << 16 );
	_width |= ( static_cast<unsigned char>( data[_startSequenceSize+9] ) << 24 );

	_height = 0;

	_height = static_cast<unsigned char>( data[_startSequenceSize+10] );
	_height |= ( static_cast<unsigned char>( data[_startSequenceSize+11] ) << 8 );
	_height |= ( static_cast<unsigned char>( data[_startSequenceSize+12] ) << 16 );
	_height |= ( static_cast<unsigned char>( data[_startSequenceSize+13] ) << 24 );

	if( 0 == _width && 0 == _height )
	{
		switch( resolution )
		{
		case 0: //QVGA
			_width = 320;
			_height = 240;
			break;

		case 1: //VGA
			_width = 640;
			_height = 480;
			break;

		default: //CustomResolution
			break;
		}
	}

  _numColorChannels = static_cast<unsigned char>( data[_startSequenceSize+14] );
  _bitsPerChannel = static_cast<unsigned char>( data[_startSequenceSize+15] );

  return true;
}

bool ImageServer::findStopSequence( const QByteArray& data )
{
  if( data.size() < _stopSequenceSize )
  {
    return false;
  }

  int n = strncmp( (const char*)(data.constData() + data.size() - _stopSequenceSize), (const char*)_stopSequence, _stopSequenceSize );
  return ( n == 0 );
}
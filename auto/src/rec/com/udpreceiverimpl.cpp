//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/com/udpreceiverimpl.h"
#include "rec/com/protocol.h"
#include "rec/core_lt/Exception.h"
#include "rec/core_lt/ByteArray.h"
#include "rec/core_lt/memory/DataStream.h"
#include <boost/bind.hpp>
#include <iostream>

using rec::com::Package;
using rec::com::PackageContainer;
using rec::com::UdpReceiverImpl;
using rec::com::Protocol;
using rec::core_lt::ByteArray;
using rec::core_lt::memory::DataStream;

Package::Package()
: _status ( UNUSED )
, _buffer ( NULL )
{
}

Package::~Package()
{
	if( _buffer != NULL )
		delete[] _buffer;
}

void Package::create( boost::uint32_t index, boost::uint32_t size )
{
	_index = index;
	_dataWritten = 0;
	_dataSize = size;

	// Reuse the buffer if reasonable
	if( _buffer == NULL || _bufferCapacity < _dataSize || _bufferCapacity > 2 * _dataSize )
	{
		if( _buffer != NULL )
			delete[] _buffer;

		_buffer = new boost::uint8_t[ _dataSize ];
		_bufferCapacity = _dataSize;
	}

	_status = RECEIVING;
}

boost::uint32_t Package::getIndex()
{
	return _index;
}

Package::Status Package::getStatus()
{
	return _status;
}

boost::uint32_t Package::getDataSize()
{
	return _dataSize;
}

void Package::addData( boost::uint8_t* data, boost::uint32_t offset, boost::uint32_t size )
{
	if( _status != RECEIVING )
		throw rec::core_lt::Exception( "Package: Not in Receving state" );

	if( offset + size > _bufferCapacity )
		throw rec::core_lt::Exception( "Package: Not enough capacity" );

	memcpy( _buffer + offset, data, size );

	_dataWritten += size;

	if( ( _dataWritten == _dataSize ) )
		_status = COMPLETE;
}

boost::uint8_t* Package::getPackageData()
{
	if( _status != COMPLETE )
		throw rec::core_lt::Exception( "Package not complete" );

	_status = UNUSED;
	return _buffer;
}

PackageContainer::PackageContainer( int numPackages )
: _packages ( new Package[ numPackages ] )
, _numPackages ( numPackages )
{
}

PackageContainer::~PackageContainer()
{
	delete[] _packages;
}

Package* PackageContainer::getPackage( int index )
{
	// Check if the package is already in the list
	for( boost::uint32_t i = 0; i < _numPackages; i++ )
	{
		if( _packages[i].getIndex()== index )
		{
			return &(_packages[i]);
		}
	}

	return NULL;
}

Package* PackageContainer::createPackage( int index, int size )
{
	Package* currentPackage = NULL;

	// Try to find an unused package
	for( boost::uint32_t i = 0; i < _numPackages; i++ )
	{
		if( _packages[i].getStatus() == Package::UNUSED )
		{
			currentPackage = &(_packages[i]);
			break;
		}
	}

	if(currentPackage == NULL)
	{
		// All packages are used. Overwrite the one with the smallest index (= the oldest one)
		currentPackage = &(_packages[0]);
		for( boost::uint32_t i = 1; i < _numPackages; i++ )
		{
			if( _packages[i].getIndex() < currentPackage->getIndex() )
			{
				currentPackage = &(_packages[i]);
			}
		}
	}

	currentPackage->create( index, size );
	return currentPackage;
}

UdpReceiverImpl::UdpReceiverImpl()
: _packageContainer( 3 )
, _io_service ( )
, _socket ( _io_service )
, _receiveThread( NULL )
, _recvBuffer( NULL )
{

}

UdpReceiverImpl::~UdpReceiverImpl()
{
	stop();
}

void UdpReceiverImpl::setCallback( boost::function< void( const boost::uint8_t *data, const boost::uint32_t dataLength ) > callback )
{
	boost::mutex::scoped_lock l( _callbackMutex );

	_callback = callback;
}

void UdpReceiverImpl::start( boost::uint16_t port )
{
	boost::mutex::scoped_lock l( _receiveThreadMutex );

	if( _receiveThread != NULL )
		throw rec::core_lt::Exception( "UdpReceiver already started" );

	running = true;

	_socket.open( boost::asio::ip::udp::v4() );
	_socket.bind( boost::asio::ip::udp::endpoint( boost::asio::ip::udp::v4(), port ) );

	_recvBuffer = new boost::uint8_t[ Protocol::MaxUdpFrameSize ];

	receive();

	_receiveThread = new boost::thread( boost::bind( &boost::asio::io_service::run, &_io_service ) );
}

void UdpReceiverImpl::stop()
{
	boost::mutex::scoped_lock l( _receiveThreadMutex );

	if( _receiveThread == NULL )
		return;

	running = false;

	_socket.shutdown( boost::asio::ip::udp::socket::shutdown_receive );

	//_receiveThread->join();
	delete _receiveThread;
	_receiveThread = NULL;

	_socket.close();

	delete _recvBuffer;
	_recvBuffer = NULL;
}

void UdpReceiverImpl::receive()
{
	_socket.async_receive_from(
		boost::asio::buffer( _recvBuffer, Protocol::MaxUdpFrameSize )
		,_remoteEndpoint
		, boost::bind( &UdpReceiverImpl::handleData, this, boost::asio::placeholders::bytes_transferred )
		);
}

// Only called by the listener thread whenever new data has been received and therefore not thread safe
void UdpReceiverImpl::handleData( std::size_t receivedBytes )
{
	if( !running )
		return;

	if( !callbackRegistered() )
		return;

	ByteArray byteArray = ByteArray::fromRawData( _recvBuffer, receivedBytes );
	DataStream stream( byteArray );

	// Check start sequence
	boost::uint32_t startSequence;
	stream >> startSequence;
	if( startSequence != rec::com::Protocol::StartSequence )
		return;

	// Read header
	boost::uint32_t curPackageIndex;
	boost::uint32_t packageSize;
	boost::uint32_t packageDataOffset;
	boost::uint32_t frameSize;

	stream >> curPackageIndex;
	stream >> packageSize;
	stream >> packageDataOffset;
	stream >> frameSize;

	// Read content
	Package* currentPackage = _packageContainer.getPackage( curPackageIndex );
	if( currentPackage == NULL )
		currentPackage = _packageContainer.createPackage( curPackageIndex, packageSize );

	boost::uint8_t temp[ Protocol::MaxUdpFrameSize ];
	stream.readRawData( temp, frameSize );
	currentPackage->addData( temp, packageDataOffset, frameSize );

	// Check stop sequence
	boost::uint32_t stopSequence;
	stream >> stopSequence;
	if( stopSequence != rec::com::Protocol::StopSequence )
		std::cerr << "UDP receiver: Invalid stop sequence" << std::endl;

	// Check if the package is complete
	if( currentPackage->getStatus() == Package::COMPLETE )
	{
		boost::uint32_t dataSize = currentPackage->getDataSize();
		boost::uint8_t* data = currentPackage->getPackageData();

		//boost::mutex::scoped_lock l( _callbackMutex );
		_callback( data, dataSize );
	}

	receive();
}

bool UdpReceiverImpl::callbackRegistered()
{
	boost::mutex::scoped_lock l( _callbackMutex );

	if( !_callback )
		return false;
	else
		return true;
}

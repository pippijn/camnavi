//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include <rec/com/udpsenderimpl.h>
#include "rec/core_lt/ByteArray.h"
#include "rec/core_lt/memory/DataStream.h"
#include <rec/com/protocol.h>
#include <iostream>

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

using rec::com::UdpSenderImpl;
using rec::com::Protocol;
using rec::core_lt::ByteArray;
using rec::core_lt::memory::DataStream;

UdpSenderImpl::UdpSenderImpl()
: _buffer( new ByteArray( Protocol::MaxUdpFrameSize ) )
, _curImageIndex( 0 )
, _io_service()
, _socket(_io_service )
{
	_socket.open( boost::asio::ip::udp::v4() );
}

UdpSenderImpl::~UdpSenderImpl()
{
	delete _buffer;
}

bool UdpSenderImpl::addDestination( const std::string& ip, boost::uint16_t port )
{
	boost::mutex::scoped_lock l( _destinationMutex );

	// Check if destination already contains this address
	std::list< boost::asio::ip::udp::endpoint >::iterator iter = _destination.begin();
	while( iter != _destination.end() )
	{
		if( (*iter).port() == port && (*iter).address().to_string() == ip )
		{
			return false;
		}
		iter++;
	}

	_destination.push_back( boost::asio::ip::udp::endpoint( boost::asio::ip::address::from_string( ip ), port ) );
	return true;
}

bool UdpSenderImpl::removeDestination( const std::string& ip, boost::uint16_t port )
{
	boost::mutex::scoped_lock l( _destinationMutex );

	std::list< boost::asio::ip::udp::endpoint >::iterator iter = _destination.begin();
	while( iter != _destination.end() )
	{
		if( (*iter).port() == port && (*iter).address().to_string() == ip )
		{
			_destination.erase( iter );
			return true;
		}
		iter++;
	}
	return false;
}

boost::uint32_t UdpSenderImpl::numDestinations()
{
	boost::mutex::scoped_lock l( _destinationMutex );

	return _destination.size();
}

void UdpSenderImpl::clearDestinations()
{
	boost::mutex::scoped_lock l( _destinationMutex );

	_destination.clear();
}

bool UdpSenderImpl::send( const unsigned char* data, boost::uint32_t dataSize )
{
	const boost::uint32_t availableFrameSize = Protocol::MaxUdpFrameSize - ( Protocol::HeaderSize + 8 /* Start- & Stopsequence */ );

	for( boost::uint32_t dataSent = 0; dataSent < dataSize; )
	{
		const boost::uint32_t dataToSent = min( availableFrameSize, dataSize - dataSent );

		DataStream stream( _buffer );

		//Add the start sequence
		stream << rec::com::Protocol::StartSequence;

		//Write the header
		stream << _curImageIndex;
		stream << dataSize;
		stream << dataSent;
		stream << dataToSent;

		//Write the actual image data
		stream.enc( data + dataSent, dataToSent );
		dataSent += dataToSent;

		//Add the stop sequence
		stream << rec::com::Protocol::StopSequence;

		//Send this frame to all registered receivers
		sendFrameToDestinations( _buffer->constData(), _buffer->size() );
	}

	_curImageIndex++;

	return true;
}

void UdpSenderImpl::sendFrameToDestinations( const unsigned char* data, boost::uint32_t dataSize )
{
	// Send a frame to all registered destinations
	boost::mutex::scoped_lock l( _destinationMutex );

	std::list< boost::asio::ip::udp::endpoint >::iterator iter = _destination.begin();
	while( iter != _destination.end() )
	{
		_socket.send_to( boost::asio::buffer( data, dataSize ), *iter );

		iter++;
	}
}

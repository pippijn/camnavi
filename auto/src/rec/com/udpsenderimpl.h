//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/memory/ByteArray.h"
#include <boost/thread/mutex.hpp>
#include <boost/asio.hpp>

namespace rec
{
	namespace com
	{
		class UdpSenderImpl
		{
		public:
			UdpSenderImpl();
			virtual ~UdpSenderImpl();

			bool addDestination( const std::string& ip, boost::uint16_t port );

			bool removeDestination( const std::string& ip, boost::uint16_t port );

			boost::uint32_t numDestinations();

			void clearDestinations();

			bool send( const unsigned char* data, boost::uint32_t dataSize );

		private:
			void sendFrameToDestinations( const unsigned char* data, boost::uint32_t dataSize );

			boost::uint32_t _curImageIndex;
			rec::core_lt::memory::ByteArray *_buffer;

			boost::asio::io_service _io_service;
			boost::asio::ip::udp::socket _socket;

			std::list< boost::asio::ip::udp::endpoint > _destination;
			boost::mutex _destinationMutex;
		};
	}
}

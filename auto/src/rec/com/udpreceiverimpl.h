//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CAMERA_COM_UDPRECEIVERIMPL_H_
#define _REC_CAMERA_COM_UDPRECEIVERIMPL_H_

#include <boost/function.hpp>
#include <boost/cstdint.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>

namespace rec
{
	namespace com
	{
		/**
		* This class is used by the UdpReceiver to represent a package, which is being received at the moment.
		* Multiple UDP frames may be needed to complete this package
		*/
		class Package
		{
		public:
			enum Status { UNUSED, RECEIVING, COMPLETE };

			Package();
			~Package();

			void create( boost::uint32_t index, boost::uint32_t size );

			boost::uint32_t getIndex();
			Status getStatus();
			boost::uint32_t getDataSize();

			void addData( boost::uint8_t* data, boost::uint32_t offset, boost::uint32_t size );

			boost::uint8_t* getPackageData();

		private:
			boost::uint32_t _index;

			boost::uint32_t _dataWritten;
			boost::uint32_t _dataSize;

			boost::uint8_t *_buffer;
			boost::uint32_t _bufferCapacity;

			Status _status;
		};

		/**
		* This class is used by the UdpReceiver to hold a number of packages, which are being received at the moment.
		*/
		class PackageContainer
		{
		public:
			PackageContainer( int numPackages );
			~PackageContainer();

			Package* getPackage( int index );
			Package* createPackage( int index, int size );

		private:
			Package *_packages;
			boost::uint32_t _numPackages;
		};

		class UdpReceiverImpl
		{
		public:
			UdpReceiverImpl();
			~UdpReceiverImpl();

			void setCallback( boost::function< void( const boost::uint8_t *data, const boost::uint32_t dataLength ) > callback );

			void start( boost::uint16_t port );
			void stop();

		private:
			void receive();
			void handleData( std::size_t receivedBytes );
			bool callbackRegistered();

			PackageContainer _packageContainer;

			boost::asio::io_service _io_service;
			boost::asio::ip::udp::socket _socket;
			boost::asio::ip::udp::endpoint _remoteEndpoint;
			boost::uint8_t *_recvBuffer;

			boost::thread *_receiveThread;
			boost::mutex _receiveThreadMutex;

			bool running;

			boost::function< void( const boost::uint8_t *data, const boost::uint32_t dataLength ) > _callback;
			boost::mutex _callbackMutex;
		};
	}
}

#endif

//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include <rec/com/udpreceiver.h>
#include <rec/com/udpreceiverimpl.h>

using rec::com::UdpReceiver;
using rec::com::UdpReceiverImpl;

UdpReceiver::UdpReceiver()
  : _impl ( new UdpReceiverImpl() )
{
}

UdpReceiver::~UdpReceiver()
{
  delete _impl;
}

void UdpReceiver::setCallback( boost::function< void( const boost::uint8_t *data, const boost::uint32_t dataLength ) > callback )
{
  _impl->setCallback( callback );
}

void UdpReceiver::start( boost::uint16_t port )
{
  _impl->start( port );
}

void UdpReceiver::stop()
{
  _impl->stop();
}

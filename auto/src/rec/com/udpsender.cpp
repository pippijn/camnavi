//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include <rec/com/udpsender.h>
#include <rec/com/udpsenderimpl.h>

using rec::com::UdpSender;

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

using rec::com::UdpSenderImpl;

UdpSender::UdpSender()
 : _impl( new UdpSenderImpl() )
{
}

UdpSender::~UdpSender()
{
  delete _impl;
}

bool UdpSender::addDestination( const std::string& ip, boost::uint16_t port )
{
  return _impl->addDestination( ip, port );
}

bool UdpSender::removeDestination( const std::string& ip, boost::uint16_t port )
{
  return _impl->removeDestination( ip, port ); 
}

boost::uint32_t UdpSender::numDestinations() const
{
  return _impl->numDestinations(); 
}

void UdpSender::clearDestinations()
{
  _impl->clearDestinations(); 
}

bool UdpSender::send( const unsigned char* data, unsigned int dataSize )
{
  return _impl->send( data, dataSize ); 
}

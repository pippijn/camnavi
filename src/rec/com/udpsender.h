//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CAMERA_COM_UDPSENDER_H_
#define _REC_CAMERA_COM_UDPSENDER_H_

#include "rec/export.h"
#include "rec/core_lt/memory/ByteArray.h"
#include <boost/cstdint.hpp>

#include <string>

namespace rec
{
namespace com
{
class UdpSenderImpl;

/**
 * Sends data of arbitrary length using UDP.
 * Packets bigger than Protocol::MaxUdpFrameSize will be split up and sent seperatly.
 * Thread safe.
 */
class REC_EXPORT UdpSender
{
public:
  UdpSender();
  virtual ~UdpSender();

  /**
   * Adds a new destination
   * 
   * @param ip The IP address of the destination
   * @param port The UDP port of the destination
   * @return True, if the destination has been added. False, if the destination already is registered.
   */
  bool addDestination( const std::string& ip, boost::uint16_t port );

  /**
   * Removes a destination
   * 
   * @param ip The IP address of the destination
   * @param port The UDP port of the destination
   * @return True, if the destination has been removed. False, if the destination isn't registered.
   */
  bool removeDestination( const std::string& ip, boost::uint16_t port );

  /**
   * Returns the number of registered destinations
   * 
   * @return The number of registered destinations
   */
  boost::uint32_t numDestinations() const;

  /**
   * Removes all destinations
   */
  void clearDestinations();

  /**
   * Sends data.
   * 
   * @param data Pointer to the data to be sent
   * @param dataSize Size of the data array
   * @return True, if successful.
   */
	bool send( const unsigned char* data, unsigned int dataSize );

protected:
  UdpSenderImpl *_impl;
};
}
}
#endif

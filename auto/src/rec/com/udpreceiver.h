//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CAMERA_COM_UDPRECEIVER_H_
#define _REC_CAMERA_COM_UDPRECEIVER_H_

#include <rec/export.h>
#include <boost/cstdint.hpp>
#include <boost/function.hpp>

namespace rec
{
namespace com
{
  class UdpReceiverImpl;

  /**
   * Receives data sent by UdpSender.
   * Thread safe.
   */
  class REC_EXPORT UdpReceiver
  {
  public:
    UdpReceiver();
    virtual ~UdpReceiver();

    void setCallback( boost::function< void( const boost::uint8_t *data, const boost::uint32_t dataLength ) > callback );

    void start( boost::uint16_t port );
    void stop();

  protected:
    UdpReceiverImpl *_impl;

  };
}
}

#endif

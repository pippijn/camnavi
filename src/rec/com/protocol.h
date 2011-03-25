//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_COM_PROTOCOL_H_
#define _REC_COM_PROTOCOL_H_

#include <boost/cstdint.hpp>

namespace rec
{
namespace com
{
class Protocol
{
public:
  static const boost::uint32_t StartSequence = 1;
  static const boost::uint32_t StopSequence = 2;

  static const boost::uint32_t HeaderSize = 16;

  static const boost::uint32_t MaxUdpFrameSize = 32768;
};
}
}

#endif

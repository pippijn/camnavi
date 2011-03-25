//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/image/jpeg.hpp"

#include <cstring>

#include <boost/lexical_cast.hpp>;

#include <Magick++.h>

rec::core_lt::image::ImageInfo
rec::core_lt::image::jpg_info (const rec::core_lt::memory::ByteArrayConst &data)
{
  throw 0;
}

rec::core_lt::image::Image
rec::core_lt::image::jpg_decompress (const rec::core_lt::memory::ByteArrayConst &data)
{
  static unsigned int imgnum;
  Magick::Image parsed (Magick::Blob (data.constData (), data.size ()));
  parsed.write ("images/" + boost::lexical_cast<std::string> (imgnum++) + ".jpg");

  //parsed.resize (Magick::Geometry (640, 480));

  Image img (ImageInfo (parsed.columns (),
                        parsed.rows (),
                        3,
                        parsed.depth (),
                        Format_BGR));

  Magick::Blob blob;
  parsed.write (&blob, "bgr");
  memcpy (img.data (), blob.data (), blob.length ());

  return img;
}


// decompression


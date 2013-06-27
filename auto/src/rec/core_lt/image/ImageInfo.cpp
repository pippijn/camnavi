//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/core_lt/image/ImageInfo.h"

std::string rec::core_lt::image::friendlyName( rec::core_lt::image::Format format )
{
	switch( format )
	{
	case Format_RGB:
		return "RGB";

	case Format_BGR:
		return "BGR";

	case Format_Gray:
		return "Gray";

	case Format_YUV:
		return "YUV";

	case Format_HSV:
		return "HSV";

	case Format_YCbCr:
		return "YCbCr";

	case Format_HLS:
		return "HLS";

	default:
		return "Undefined";
	}
}

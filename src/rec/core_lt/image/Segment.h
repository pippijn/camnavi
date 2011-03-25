//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_IMAGE_SEGMENT_H_
#define _REC_CORE_LT_IMAGE_SEGMENT_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/List.h"
#include "rec/core_lt/Point.h"
#include "rec/core_lt/Color.h"
#include "rec/core_lt/Line.h"
#include "rec/core_lt/String.h"

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			class Segment
			{
			public:
				typedef rec::core_lt::List< rec::core_lt::Line > LineContainer;

				Segment()
					: area( 0 )
					, index( 0 )
				{
				}

				unsigned int area;
				rec::core_lt::Color color;
				Point midPoint;
				LineContainer lines;
				unsigned int index;
				String name;
			};

			class SegmentList : public List< Segment >
			{
			public:
				bool operator==( const SegmentList& other ) const
				{
					if( size() != other.size() )
					{
						return false;
					}

					const_iterator i = constBegin();
					const_iterator otheri = other.constBegin();
					while( constEnd() != i )
					{
						if( (*i).color != (*otheri).color )
						{
							return false;
						}

						++otheri;
						++i;
					}

					return true;
				}

				bool operator!=( const SegmentList& other ) const
				{
					return !operator==( other );
				}
			};
		}
	}
}

#endif

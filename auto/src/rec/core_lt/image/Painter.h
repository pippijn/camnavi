//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_IMAGE_PAINTER_H_
#define _REC_CORE_LT_IMAGE_PAINTER_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/image/Image.h"

#include "rec/core_lt/Color.h"
#include "rec/core_lt/Point.h"
#include "rec/core_lt/Line.h"

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			class REC_CORE_LT_EXPORT Painter
			{
			public:
				Painter( Image* image );
				Painter();

				void begin( Image* image );

				void end();

				void setPen( const Color& color, unsigned int width );
				
				void setBrush( const Color& color );

				void drawLine( int x0, int y0, int x1, int y1, const Color& color = Color::null );

				void drawLine( const Line& line, const Color& color = Color::null );

				void drawLine( const Point& p0, const Point& p1, const Color& color = Color::null );
				
				void drawLine( const Point& p0, float phi, unsigned int length );

				void drawPoint( int x, int y, const Color& color = Color::null );
				void drawPoint( const Point& p, const Color& color = Color::null );

				void drawCircle( const Point& mid, unsigned int radius );

				void drawCircle( int midx, int midy, unsigned int radius );

				///Fill image with color
				void fill( const Color& color );

			private:
				void drawPoint_i( int x, int y, const Color& color = Color::null );

				Image* _image;

				Color _penColor;
				unsigned int _penWidth;
				
				Color _brushColor;
			};
		}
	}
}

#endif //_REC_CORE_LT_IMAGE_PAINTER_H_


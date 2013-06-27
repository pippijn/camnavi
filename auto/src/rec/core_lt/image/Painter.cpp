//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/image/Painter.h"

#include <algorithm>

using namespace rec::core_lt;
using namespace rec::core_lt::image;

Painter::Painter( Image* image )
: _image( image )
, _penWidth( 1 )
, _brushColor( Color::null )
{
}

Painter::Painter()
: _image( NULL )
, _penWidth( 1 )
, _brushColor( Color::null )
{
}

void Painter::begin( Image* image )
{
	_image = image;
}

void Painter::end()
{
	_image = NULL;
}

void Painter::setPen( const Color& color, unsigned int width )
{
	_penColor = color;
	_penWidth = width;
}

void Painter::setBrush( const Color& color )
{
	_brushColor = color;
}

void Painter::drawLine( int x0, int y0, int x1, int y1, const Color& color )
{
	if( !_image )
	{
		return;
	}

	//Bresenham's algorithm
	bool steep = ( abs( y1 - y0 ) > abs( x1 - x0 ) );

	int tmp;

  if( steep )
	{
		tmp = x0;
		x0 = y0;
		y0 = tmp;

		tmp = x1;
		x1 = y1;
		y1 = tmp;
	}

  if( x0 > x1 )
	{
		tmp = x1;
		x1 = x0;
		x0 = tmp;

		tmp = y1;
		y1 = y0;
		y0 = tmp;
	}

  int deltax = x1 - x0;
	int deltay = abs(y1 - y0);
  float error = 0.0;
  float deltaerr = static_cast<float>( deltay ) / static_cast<float>( deltax );
  int ystep;
  int y = y0;
  if( y0 < y1 )
	{
		ystep = 1;
	}
	else
	{
		ystep = -1;
	}
  
	for( int x=x0; x<=x1; x++ )
	{
		if( steep )
		{
			drawPoint( y, x, color);
		}
		else
		{
			drawPoint( x, y, color );
		}

		error = error + deltaerr;
		if( error >= 0.5 )
		{
			y = y + ystep;
			error = error - 1.0f;
		}
	}
}

void Painter::drawLine( const Point& p0, const Point& p1, const Color& color )
{
	drawLine( p0.x(), p0.y(), p1.x(), p1.y(), color );
}

void Painter::drawLine( const Line& line, const Color& color )
{
	drawLine( line.x0(), line.y0(), line.x1(), line.y1(), color );
}

void Painter::drawPoint( int x, int y, const Color& color )
{
	if( 0 == _penWidth )
	{
		return;
	}

	int d = static_cast<int>( _penWidth / 2 );

	for( int i = -d; i<=d; i++ )
	{
		for( int j = -d; j<=d; j++ )
		{
			drawPoint_i( x+i, y+j, color );
		}
	}
}

void Painter::drawPoint_i( int x, int y, const Color& color )
{
	if( !_image )
	{
		return;
	}

	if( x < 0 || y < 0 || x >= static_cast<int>( _image->info().width ) || y >= static_cast<int>( _image->info().height ) )
	{
		return;
	}

	Color pen;
	if( color.isNull() )
	{
		pen = _penColor;
	}
	else
	{
		pen = color;
	}

	if( Format_Gray == _image->info().format )
	{
		_image->data()[ _image->step() * y + x ] = pen.grey();
	}
	else
	{
		_image->data()[ _image->step() * y + x * _image->info().numChannels     ] = pen.r();
		_image->data()[ _image->step() * y + x * _image->info().numChannels + 1 ] = pen.g();
		_image->data()[ _image->step() * y + x * _image->info().numChannels + 2 ] = pen.b();
	}
}

void Painter::drawPoint( const Point& p, const Color& color )
{
	drawPoint( p.x(), p.y(), color );
}

void Painter::drawCircle( const Point& mid, unsigned int radius )
{
	drawCircle( mid.x(), mid.y(), radius );
}

void Painter::drawCircle( int x0, int y0, unsigned int radius )
{
	//Midpoint circle algorithm

	int f = 1 - radius;
	int ddF_x = 0;
	int ddF_y = -2 * radius;
	int x = 0;
	int y = radius;

	unsigned int penWidth = _penWidth;

	if( !_brushColor.isNull() )
	{
		_penWidth = 1;
		for( int i = -y; i<=y; i++ )
		{
			drawPoint( x0, y0 - i, _brushColor );
		}
		_penWidth = penWidth;
	}

	drawPoint( x0, y0 + radius, _penColor );
	drawPoint( x0, y0 - radius, _penColor );
	drawPoint( x0 + radius, y0, _penColor );
	drawPoint( x0 - radius, y0, _penColor );


	while(x < y) 
	{
		if(f >= 0) 
		{
			y--;
			ddF_y += 2;
			f += ddF_y;
		}
		x++;
		ddF_x += 2;
		f += ddF_x + 1;

		if( !_brushColor.isNull() )
		{
			_penWidth = 1;
			for( int i = -y; i<=y; i++ )
			{
				drawPoint( x0 + x, y0 - i, _brushColor );
				drawPoint( x0 - x, y0 - i, _brushColor );
			}
			_penWidth = penWidth;
		}

		drawPoint( x0 + x, y0 + y, _penColor );
		drawPoint( x0 - x, y0 + y, _penColor );
		drawPoint( x0 + x, y0 - y, _penColor );
		drawPoint( x0 - x, y0 - y, _penColor );
		drawPoint( x0 + y, y0 + x, _penColor );
		drawPoint( x0 - y, y0 + x, _penColor );
		drawPoint( x0 + y, y0 - x, _penColor );
		drawPoint( x0 - y, y0 - x, _penColor );
	}
}

void Painter::fill( const Color& color )
{
	if( !_image )
	{
		return;
	}

	if( Format_Gray == _image->info().format )
	{
		boost::uint8_t grey = color.grey();

		for( unsigned int y=0; y<_image->info().height; y++ )
		{
			for( unsigned int x=0; x<_image->info().width; x++ )
			{
				_image->data()[ _image->step() * y + x ] = grey;
			}
		}
	}
	else
	{
		for( unsigned int y=0; y<_image->info().height; y++ )
		{
			for( unsigned int x=0; x<_image->info().width; x++ )
			{
				_image->data()[ _image->step() * y + x * _image->info().numChannels     ] = color.r();
				_image->data()[ _image->step() * y + x * _image->info().numChannels + 1 ] = color.g();
				_image->data()[ _image->step() * y + x * _image->info().numChannels + 2 ] = color.b();
			}
		}
	}
}

void Painter::drawLine( const Point& p0, float phi, unsigned int length )
{
	int x2 = p0.x() + static_cast<int>( cos( 0.0175 * phi ) * length );
	int y2 = p0.y() + static_cast<int>( sin( 0.0175 * phi ) * length );

	drawLine( p0.x(), p0.y(), x2, y2 );
}

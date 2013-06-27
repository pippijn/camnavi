/**
 * This file is part of the OpenVIDIA project at http://openvidia.sf.net
 * Copyright (C) 2004, James Fung
 *
 * OpenVIDIA is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * OpenVIDIA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OpenVIDIA; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 **/

#include <openvidia/openvidia32.h>
#include <errno.h>
#include <assert.h>


#include <iostream>
using namespace std;

MemoryImage::MemoryImage(void *data, int w, int h, int comp, GLenum fmt) : 
    LBuffer(w,h,comp) 
{
    if( data != NULL ) {
        memcpy( ptr(), data, w*h*comp );
    }

    //glGenFramebuffersEXT(1,&fbo); 
    //glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,fbo);

	glGenTextures(1, &texObj);              // texture 	 
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texObj);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV,
	                GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV,
	                GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_REPLACE );

    // this should actually do all cases.
    assert( fmt == GL_RGBA || fmt == GL_FLOAT_RGBA32_NV || fmt == GL_FLOAT_RGBA16_NV );
    switch ( fmt ) {
        case GL_RGBA : 
	        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, fmt, 
	                width(), height(), 0,
	                GL_RGBA, GL_UNSIGNED_BYTE,ptr() );
            break;
        case GL_FLOAT_RGBA32_NV :
        case GL_FLOAT_RGBA16_NV :
            glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, fmt,
                    width(), height(), 0,
                    GL_RGBA, GL_FLOAT, ptr() ) ;  
            break;
        default :
            cerr<<"MemoryImage:: Could not initialze texture based on your data type."<<endl;
            break;
    }
	dirtyBit = true;
       //glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, 
       //                           GL_COLOR_ATTACHMENT0_EXT, 
       //                           GL_TEXTURE_RECTANGLE_NV,texObj,0);
    //glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);
}

void *MemoryImage::ptr()
{
    return LBuffer::ptr();
}

/**Return the width of the image
 * @return The width of the image.
 */
///Width of the image.
GLuint MemoryImage::w() 
{
	return width();
}
/**Return the height of the image
 * @return The height of the image.
 */
 ///Height of the image.
GLuint MemoryImage::h() 
{
	return height();
}

/**Returns the texture handle holding the image.  The texture is
 * a GL_TEXTURE_RECTANGLE_NV, holding GL_RGBA values.
 * @return The texture handle holding the image.
 */
///Get texture handle of the image and set the texture as clean (used).
GLuint MemoryImage::tex() 
{
	dirtyBit = false;
	//dirtyBit = true;
	//cerr<<"Returning File Texture"<<endl;
	return texObj;
}
/**Returns the texture handle holding the image.  The texture is
 * a GL_TEXTURE_RECTANGLE_NV, holding GL_RGBA values.
 * @return The texture handle holding the image.
 */
 ///Get texture handle of the image widhout setting dirty bit.
GLuint MemoryImage::tex_nodirty()
{
	return texObj;
}

void MemoryImage::resetDirty(){
	dirtyBit = true;
}

/**Check the dirty bit of this texture.
 * @return TRUE if the texture is dirty (has been changed and has not been used), FALSE if the
 * 			texture has already been processed at least once.
 */
///Query the dirty bit (false = texture has been used/processed, true = texture has not been processed.).
bool MemoryImage::dirty()
{
    return dirtyBit;
}

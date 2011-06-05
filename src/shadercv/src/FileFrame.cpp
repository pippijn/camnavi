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
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <errno.h>
#include <assert.h>
#include <Imlib2.h>


#include <iostream>
#include <map>
using namespace std;

map<string, Imlib_Image> img;

/**Load an image from disk.
 * @return Returns 1 on success, -1 on failure.
 */
int FileFrame::load_image( char *fname ) 
{
   if (img[fname])
     {
     Im = img[fname];
     imlib_context_set_image(Im);
     return 1;
     }
   fprintf(stderr,"Opening %s\n", fname );

   img[fname] = Im  = imlib_load_image(fname );
   imlib_context_set_image(Im);
	
   if( Im == NULL ) {
     printf("Could not load the image\n");
     return -1;
   }

   return 1;
}

/**Loads a file from disk.  FileFrame supports the img_src interface and 
 * can be used as an image source.  Additionally, it is a LBuffer, so supports
 * the LBuffer interface for memory access and locking.
 * Internally, this creates an OpenGL texture holding the image.  Thus, FileFrame() objects
 * should only be created after initializing the OpenGL contexts.
 * @param fname The filename of the file to load from disk.  The file must be an image,
 *              and can be of any type supported by Imlib (image loading library).
 */
 ///Creates a file object, loading an image from disk.
FileFrame::FileFrame(char *fname) : LBuffer(1,1,4) , Im (0)
{
  load (fname);
}

void
FileFrame::load (char const *fname)
{
  load_image((char*)fname);
  static int o;
  if (!o)
    {
      o = 1;
      resize( imlib_image_get_width(),  imlib_image_get_height(), 4 );
      glGenTextures(1, &texObj);              // texture 	 
    }
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, texObj);
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV,
                  GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV,
                  GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_REPLACE );
  glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA, 
               imlib_image_get_width(), imlib_image_get_height(), 0,
               GL_BGRA, GL_UNSIGNED_BYTE,imlib_image_get_data() );
  dirtyBit = true;
}

/**ptr returns a pointer to the imlib loaded data. This overrides the default
 * ptr function from LBuffer
 * @return pointer to the memory.  The memory is unsigned char type, RGBA.
 */
void * FileFrame::ptr() 
{
	imlib_context_set_image(Im);
	return imlib_image_get_data();
}
/*
char * FileFrame::getComponent()
{
    static char *componentMemory;

	imlib_context_set_image(Im);
	char *data = (char *)imlib_image_get_data();

    for( int i=0; i<width*height*4 ; i++ ) { 
    }
}
*/

/**Return the width of the image
 * @return The width of the image.
 */
///Width of the image.
GLuint FileFrame::w() 
{
	return width();
}
/**Return the height of the image
 * @return The height of the image.
 */
 ///Height of the image.
GLuint FileFrame::h() 
{
	return height();
}

/**Returns the texture handle holding the image.  The texture is
 * a GL_TEXTURE_RECTANGLE_NV, holding GL_RGBA values.
 * @return The texture handle holding the image.
 */
///Get texture handle of the image and set the texture as clean (used).
GLuint FileFrame::tex() 
{
  static int i = 0;
  if (i > 390)
    i = 0;
  char buf[20];
  sprintf (buf, "images/%03d.jpg", i++);
  load (buf);
	//dirtyBit = false;
        //dirtyBit = true;
        //cerr<<"Returning File Texture"<<endl;
	return texObj;
}
/**Returns the texture handle holding the image.  The texture is
 * a GL_TEXTURE_RECTANGLE_NV, holding GL_RGBA values.
 * @return The texture handle holding the image.
 */
 ///Get texture handle of the image widhout setting dirty bit.
GLuint FileFrame::tex_nodirty()
{
	return texObj;
}

void FileFrame::resetDirty(){
	dirtyBit = true;
}

/**Check the dirty bit of this texture.
 * @return TRUE if the texture is dirty (has been changed and has not been used), FALSE if the
 * 			texture has already been processed at least once.
 */
///Query the dirty bit (false = texture has been used/processed, true = texture has not been processed.).
bool FileFrame::dirty()
{
    return dirtyBit;
}

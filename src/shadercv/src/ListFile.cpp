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
using namespace std;

string ListFile::getNextFilename() 
{
  string line ;
  if (myfile.is_open())
  {
    if ( myfile.eof() )
    {
        myfile.close();
        myfile.open(listFileName.c_str() );
    if ( myfile.eof() )
    {
      cerr<<"Error: empty listfile"<<endl;
      exit(0);
    }
    }
    getline (myfile,line);
    cout << line << endl;
  }
  else {
    cout << "Unable to open list file"<<endl; 
    }
  return line;
}

int ListFile::openListFile( char *fname ) 
{
  //ifstream myfile(fname);
   myfile.open(fname);
  
  if (myfile.is_open()) {
    cout<<"Opened "<<fname<<endl;
    return 0;
  }
  else {
    cout << "Unable to open file: " << fname<<endl; 
    return -1;
  }
}

ListFile::ListFile( char *fname ) 
{
    listFileName = string(fname);
    ff = NULL;
    openListFile(fname);    
    advanceFile();
}

string ListFile::advanceFile() 
{
    string s = getNextFilename(); 
    if( s.length() != 0 ) {
        if( ff == NULL ) ff = new FileFrame( (char*)s.c_str() ); 
        else  ff->load_image( (char *)s.c_str() );
        return s;
    } else {
	    cerr<<"Unable to Open "<<s<<endl;
        return s;
    }
}

GLuint ListFile::tex() 
{
    advanceFile();
	return ff->tex();
}

/**ptr returns a pointer to the imlib loaded data. This overrides the default
 * ptr function from LBuffer
 * @return pointer to the memory.  The memory is unsigned char type, RGBA.
 */
void * ListFile::ptr() 
{
    if( ff == NULL ) return NULL;
	return ff->ptr();
}

/**Return the width of the image
 * @return The width of the image.
 */
///Width of the image.
GLuint ListFile::w() 
{
    if( ff == NULL ) return 0;
	return ff->w();
}
/**Return the height of the image
 * @return The height of the image.
 */
 ///Height of the image.
GLuint ListFile::h() 
{
    if( ff == NULL ) return 0;
	return ff->h();
}

/**Returns the texture handle holding the image.  The texture is
 * a GL_TEXTURE_RECTANGLE_NV, holding GL_RGBA values.
 * @return The texture handle holding the image.
 */
 ///Get texture handle of the image widhout setting dirty bit.
GLuint ListFile::tex_nodirty()
{
    if( ff == NULL ) return 0;
	return ff->tex_nodirty();
}

void ListFile::resetDirty(){
    if( ff == NULL ) return ;
    ff->resetDirty();
}

/**Check the dirty bit of this texture.
 * @return TRUE if the texture is dirty (has been changed and has not been used), FALSE if the
 * 			texture has already been processed at least once.
 */
///Query the dirty bit (false = texture has been used/processed, true = texture has not been processed.).
bool ListFile::dirty()
{
    if( ff == NULL ) return true;
    return ff->dirty();
}

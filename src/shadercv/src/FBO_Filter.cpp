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

#ifdef WIN32
#include <GL/glew.h>
#define GLEW_STATIC 1
#endif

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <stdio.h>
#include <assert.h>
#include <Cg/cgGL.h>
#include <openvidia/openvidia32.h>
#define BUFFER_OFFSET(i) ((char *)NULL + (i))	


#define ERRCHECK() \

#define CHECK_FRAMEBUFFER_STATUS() \
{\
 GLenum status; \
 status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT); \
 switch(status) { \
 case GL_FRAMEBUFFER_COMPLETE_EXT: \
   /*fprintf(stderr,"framebuffer COMPELTE\n");*/\
   break; \
 case GL_FRAMEBUFFER_UNSUPPORTED_EXT: \
   fprintf(stderr,"framebuffer GL_FRAMEBUFFER_UNSUPPORTED_EXT\n");\
    /* you gotta choose different formats */ \
   assert(0); \
   break; \
 case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT: \
   fprintf(stderr,"framebuffer INCOMPLETE_ATTACHMENT\n");\
   break; \
 case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT: \
   fprintf(stderr,"framebuffer FRAMEBUFFER_MISSING_ATTACHMENT %d\n",__LINE__);\
   break; \
 case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT: \
   fprintf(stderr,"framebuffer FRAMEBUFFER_DIMENSIONS\n");\
   break; \
 case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT: \
   fprintf(stderr,"framebuffer INCOMPLETE_FORMATS\n");\
   break; \
 case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT: \
   fprintf(stderr,"framebuffer INCOMPLETE_DRAW_BUFFER %s:%d\n",__FILE__,__LINE__);\
   break; \
 case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT: \
   fprintf(stderr,"framebuffer INCOMPLETE_READ_BUFFER %s:%d\n",__FILE__,__LINE__);\
   break; \
 case GL_FRAMEBUFFER_BINDING_EXT: \
   fprintf(stderr,"framebuffer BINDING_EXT\n");\
   break; \
\
/*
 case GL_FRAMEBUFFER_STATUS_ERROR_EXT: \
   fprintf(stderr,"framebuffer STATUS_ERROR\n");\
   break; \

 this one got left out of /usr/include/GL/glext.h v7667 nvidia drivers?\
*/\
 default: \
   /* programming error; will fail on all hardware */ \
   assert(0); \
 }\
}

static GLenum texUnitIDS[8] = {
        GL_TEXTURE0_ARB, 
        GL_TEXTURE1_ARB, 
        GL_TEXTURE2_ARB, 
        GL_TEXTURE3_ARB, 
        GL_TEXTURE4_ARB, 
        GL_TEXTURE5_ARB, 
        GL_TEXTURE6_ARB, 
        GL_TEXTURE7_ARB };

//#include <openvidia/errutil.h>

/*
 * A callback function for cg to use when it encounters an error
 */
static CGcontext errContext;

static void cgErrorCallback(void) {
    CGerror LastError = cgGetError();

    if(LastError)
    {
        const char *Listing = cgGetLastListing(errContext);
        printf("\n---------------------------------------------------\n");
        printf("%s\n\n", cgGetErrorString(LastError));
        printf("%s\n", Listing);
        printf("---------------------------------------------------\n");
        printf("Cg error, exiting...\n");
        exit(0);
    }
}


CGprogram FBO_Filter::load_cgprogram(CGprofile prof, char *name, char **args) 
{
    fprintf(stderr, "loading %s\n", name);
    return cgCreateProgramFromFile( cgContext, CG_SOURCE,
            name, prof, "FragmentProgram", (const char **)args);
}

void FBO_Filter::renderBegin() 
{
    //since we might be operating on downsampled images, we need 
    // to reshape our current drawing area. Record the previous
    // settings first though. 

    glGetIntegerv(GL_VIEWPORT, previousViewportDims );
 	// viewport for 1:1 pixel=texel mapping
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, tWidth, tHeight, 0.0 );
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glViewport(0, 0,tWidth, tHeight);
    
    //Flip the texture (BUG: No idea why the texture is flipped 
    //everytime, but this patch will just reverse that.

    glPushMatrix();
    glLoadIdentity();
    glTranslatef(tWidth, 0, 0.0f);
    glRotatef(180.0f, 0.0f, 1.0f, 0.0f);
    glTranslatef(tWidth, tHeight, 0.0f);
    glRotatef(180.0f, 0.0f, 0.0f, 1.0f);

    
    for(unsigned int i=0 ; i<srcs.size() ;  i++ ) {
		glActiveTextureARB( texUnitIDS[i] );
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, srcs[i]->tex_nodirty() );
    }
    
   
}

void FBO_Filter::setSource( img_src *s ) {
    // new source must have identical spatial dimensions.
    assert((int)s->w() == (int)tWidth );
    assert((int)s->h() == (int)tHeight );
    srcs.clear();
    srcs.push_back(s);
    source_ptr = s;
    //createTexture(s->w(), s->h(), outputPrecision );
}

// should be one of GL_UNSIGNED_BYTE, GL_FLOAT_RGBA32_NV, GL_FLOAT_RGBA16
void FBO_Filter::setOutputPrecision( GLint fmt ) { 
    outputPrecision = fmt;
    // need to recreate the internal texture/FBO to hold the new precision.
    createTexture(source_ptr->w(), source_ptr->h(), fmt ) ;

}

void FBO_Filter::resize( GLint W, GLint H ) {
  	     createTexture( W, H, outputPrecision );
  	 }

//return the rendering state to what it was before.
void FBO_Filter::renderEnd() 
{
	glPopMatrix();
    glViewport( previousViewportDims[0], previousViewportDims[1],
            previousViewportDims[2], previousViewportDims[3] );
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    

}


void FBO_Filter::onvTapCoords(float baseX, float baseY ) 
{
	float eps = -0.0;
        for( unsigned int i=0; i<taps.size() ; i++ ) {
                glMultiTexCoord2fARB(texUnitIDS[i], baseX + taps[i].x()+eps , baseY + taps[i].y()+eps );
        }
}

void FBO_Filter::drawQuadTex(int inW, int inH)
{
    glBegin(GL_QUADS);
        onvTapCoords(0.0f, 0.0f);
        glVertex2f(0.0, 0.0);
        
        
        onvTapCoords((float)inW , 0);
        glVertex2f(tWidth, 0.0);
                        
        onvTapCoords((float)inW , (float)inH);
        glVertex2f(tWidth, tHeight);
        
        onvTapCoords(0.0f, (float)inH);  
        glVertex2f(0.0, tHeight);
    glEnd();
}
///\brief Set (change) the fragment program this filter uses.
/** @param fname The filename of the fragment program.
 */
void FBO_Filter::setProgram(char *fname) 
{
    if( fname != NULL )  {
        cgProgram = load_cgprogram(cgProfile, fname, NULL );
        cgGLLoadProgram( cgProgram );
    }
    else {
        cgProgram = 0;
    }


}

void FBO_Filter::createTexture(int W, int H, GLint fmt) 
{
    if( oTex != 0 ) glDeleteTextures(1, &oTex );
    glGenTextures(1, &oTex);              // texture 

    if( fb != 0 ) glDeleteFramebuffersEXT(1, &fb ) ;
    glGenFramebuffersEXT( 1, &fb );
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, fb );
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, oTex );
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_REPLACE );
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, oTex);
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_REPLACE );
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV,
                GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV,
                GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
   glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, fmt,W,H, 0,GL_RGBA, GL_UNSIGNED_BYTE, NULL );
    tWidth = W;
    tHeight= H;


    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
        GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_NV, oTex, 0);
        CHECK_FRAMEBUFFER_STATUS()
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}
///\brief Construct a fragment program filter, rendering the result to a texture.
/**The FBO_Filter object allows you to write a Cg fragment program to
 * filter any image source (img_src) object.  A number of objects are
 * derived from the img_src object to provide image inputs.   For instance,
 * Firewire cameras are implemented in the Dc1394 object, and image files
 * can be loaded with the FileFrame object, both of which can be used as
 * img_src objects to input images to this FBO_Filter object.
 * The FBO_Filter object itself is also an img_src, which allows multiple
 * filters to be pipelined in series.
 * 
 * The filtering operation that is to be performed is determined by the 
 * Cg fragment program that this object is created with.   The filtering
 * will take place whenever this objects tex() function is called.  As with
 * all img_src objects, calling the tex() function will query all previous
 * filters/sources in the pipeline for their latest, processed image.  Thus,
 * when writing your programs it is only necessary to call tex() on the final
 * result you wish to use.
 * 
 * Be sure that OpenGL is initialized (say through initializing glut) before
 * creating any of this object type.
 * 
 * Typical usage: 
 * 
 * <b>Initialize GL</b>
 * <pre>
 *    glutInit(&argc, argv);  
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
   glutInitWindowSize(width, height );
   glutCreateWindow(argv[0]);
   </pre>
   <b>Make some sort of image source</b>
   <pre>
   CamSource = new Dc1394(width,height);
   </pre>
   
 * <b>Create the filter, hook it up to the image source</b>
 * <pre>
 * FiltUndistort = new FBO_Filter( "FP-func-undistort.cg", CamSource, GL_RGBA );
 * </pre>
 * <b>Apply the filter, retrieve the texture handle and bind it for display.</b>
 * <pre>
 * glBindTexture(GL_TEXTURE_RECTANGLE_NV, FiltRGB2HSV->tex() ) ;
 * </pre>
 * <b>Display the result:</b>
 * <pre>
 *   glBegin(GL_QUADS); 
    glTexCoord2f(0, height);
    glVertex3f(0.0, 0.0,d );
    ...
 * </pre>
 */
FBO_Filter::FBO_Filter( 
                char *name,         ///< filename of the Cg program.  Note that
                /// the entry point function should be named
                /// "FragmentProgram" in the Cg 
                img_src *src,
                GLint fmt,
                ///A vector of 2D offset coordinates for filter taps.   Up to
                ///eight filter taps are supported.  For example, a horizontal
                ///filter may wish to specify the taps (-1,0),(0,0),(+1,0)
                vector<Coords> *inTaps,
                /// (Optional) arguments to the cgc compiler. Useful for
                /// example to specify compile time #defines for example
                char **args )
{
        outputPrecision = fmt;
		fpname = NULL;
		srcs.push_back(src);
        //Load the Cg Program.
        cgProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT) ;
        cgContext = cgCreateContext();
        errContext = cgContext;

        cgSetErrorCallback(cgErrorCallback);
        if( name != NULL )  {
                cgProgram  = load_cgprogram(cgProfile, name, args );
                cgGLLoadProgram( cgProgram );
                fpname = (char *)malloc( strlen(name) + 2 );
                strcpy( fpname, name );
        }
        else {
                cgProgram = 0;
        }
        
#define CGDIR SRCDIR"/src/shadercv/shaders"
        cgDisplayProgram  = load_cgprogram(cgProfile, CGDIR"/FP-basic.cg", NULL );
        cgGLLoadProgram( cgDisplayProgram );

        oTex = 0;
        fb = 0; 
        createTexture( src->w(), src->h(), fmt );

        glGenBuffers( 1, &pboBuf);
        glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pboBuf);
        int texSize = tWidth*tHeight*4;
        if( outputPrecision == GL_FLOAT_RGBA32_NV || outputPrecision == GL_FLOAT_RGBA16_NV ) {
            texSize *= 4;
        }
        glBufferData(GL_PIXEL_PACK_BUFFER_ARB,texSize, NULL, GL_STREAM_READ);
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0); 
        glReadBuffer(GL_FRONT);
    
                
        if( inTaps == NULL ) {
            taps.push_back(Coords(0,0));
        }
        else {
            for( unsigned int i=0; i <inTaps->size() ; i++ )  {
                taps.push_back((*inTaps)[i]);
            }
        }
        source_ptr = src;
}

/** Destroy this FBO_Filter object.  Deletes the textures used by this object.
 *  Deletes the texture that would have been referenced by the tex() call on this
 * object.
 */
FBO_Filter::~FBO_Filter()
{
	if( fpname != NULL ) free (fpname);
    glDeleteTextures(1, &oTex);
    glDeleteFramebuffersEXT(1, &fb);
    
}

/** This function is mostly deprecated in favour of the tex() method.  This
 * function causes this filter to execute on its current source data.  The
 * filter always operates, and it does not check to see if its source has been
 * updated.  Optionally, this function will readback the processing results to
 * memory pointed to by the rb_buf parameter.  The calling application should
 * allocate and manage the rb_buf memory.  The readback call will be performed
 * with the format of rb_rmt, and type of rb_type.
 * 
 * @param src The img_src to act upon.
 * @param rb_buf A pointer to memory where the readback results will be written.  
 * 				This memory should be created and managed by the calling program.
 * @param rb_fmt The pixel format to readback.  For example  GL_RGBA
 * @param rb_type The data type for readback.  For example GL_UNSIGNED_BYTE would
 * 				readback unsigned bytes.
 * @return The GL texture handle of the texture containing the processed image.
 */
GLuint FBO_Filter::apply( img_src *src, GLvoid *rb_buf, GLenum fmt ){
    //cerr<<fpname <<" is about to ask for my source's tex()"<<endl;
    return apply( src->tex(), src->w(), src->h(), rb_buf, fmt );
}

///TODO figure out readback formats from target texture information.
GLuint FBO_Filter::apply( GLuint iTex,     ///< OpenGL texture object to use as input
               GLuint inW, GLuint inH, 
		                                           ///  FBO_filter.  False if not.
                GLvoid *rb_buf =NULL,
                GLenum rb_fmt = GL_RGBA
                )
{
		//cerr<<"Applying "<<fpname<<endl;
        GLint activeARB;
      
        glGetIntegerv( GL_CLIENT_ACTIVE_TEXTURE_ARB, &activeARB );


        //XXX all this frambuffer binding is suboptimal, but will work for now.
		CHECK_FRAMEBUFFER_STATUS()
		glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, fb );
        //glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
        glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

		glEnable(GL_TEXTURE_RECTANGLE_NV);

        renderBegin();
        cgGLEnableProfile(cgProfile);
        cgGLBindProgram(cgProgram);
        
        drawQuadTex(inW, inH);

        cgGLDisableProfile(cgProfile);
        renderEnd();

        
        if( rb_buf != NULL ) {

        GLenum tmp_type = GL_UNSIGNED_BYTE;
        int texSize = tWidth*tHeight*4;
        if( outputPrecision == GL_FLOAT_RGBA32_NV || outputPrecision == GL_FLOAT_RGBA16_NV ) {
            texSize *= sizeof(float);
            tmp_type = GL_FLOAT;
        }

        //glReadBuffer(attachmentpoints[readTex]);
/*
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pboBuf);
        glBufferData(GL_PIXEL_PACK_BUFFER_ARB,texSize, NULL, GL_STREAM_READ);
        glReadPixels (0, 0, tWidth, tHeight, rb_fmt, tmp_type, BUFFER_OFFSET(0));
        void* mem = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);   
        memcpy(rb_buf, mem, texSize);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0); 
*/



      	glReadPixels( 0, 0, tWidth, tHeight, rb_fmt, tmp_type, rb_buf );
        //float x = *((float *)rb_buf);
        //fprintf(stderr, "f = %f\n", x);
      	//glReadPixels( 0, 0, tWidth, tHeight, GL_RGBA, outputPrecision, rb_buf );

        }

        //glDrawBuffer(GL_BACK);
        //glReadBuffer(GL_BACK);
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);
        glActiveTextureARB( activeARB );
// XXX not necessarily back

        return oTex ;
}

bool FBO_Filter::activateViewerProgram() 
{
    if( outputPrecision == GL_FLOAT_RGBA32_NV || outputPrecision == GL_FLOAT_RGBA16_NV ) {
        cgGLEnableProfile(cgProfile);
        cgGLBindProgram(cgDisplayProgram);
        return true;
    } 
    else {
        return false;
    }
}

void FBO_Filter::deactivateViewerProgram() 
{
    if( outputPrecision == GL_FLOAT_RGBA32_NV || outputPrecision == GL_FLOAT_RGBA16_NV ) {
        cgGLDisableProfile(cgProfile);
    } 
    return;
}


///\brief Get the width of the processed image.
/**
 *  @return The width of the processed image.
 */
GLuint FBO_Filter::w() 
{
    return tWidth;
}
///\brief Get the height of the processed image.
/*
 * @return The height of the processed image.
 */
GLuint FBO_Filter::h() 
{
    return tHeight;
}

GLuint FBO_Filter::tex_nodirty()
{
	return oTex;	
}
GLuint FBO_Filter::tex() 
{
	//cerr<<"Attempting "<<fpname<<" processing.."<<endl;
    if( source_ptr->dirty() ) {
    	//cerr<< fpname <<"'s source is dirty. attempting processing"<<endl;
        apply( source_ptr );
     }
     //cerr<<fpname <<" returning result."<<endl;
    return oTex;
}

GLuint FBO_Filter::tex_apply()
{
	apply( source_ptr );
	return oTex;	
}

bool FBO_Filter::dirty()
{
    return source_ptr->dirty();
}

void FBO_Filter::resetDirty()
{
	source_ptr->resetDirty();
}

/**
 * Set the CG parameters to the value.
 * Array size must be 4.
 */
void FBO_Filter::setCGParameter(char *name, float *value )
{
  CGparameter cgp = cgGetNamedParameter(cgProgram, name);
  cgGLSetParameter4fv(cgp, value);
}
void FBO_Filter::setCGParameter(char *name, double *value )
{
  CGparameter cgp = cgGetNamedParameter(cgProgram, name);
  cgGLSetParameter4dv(cgp, value);
}


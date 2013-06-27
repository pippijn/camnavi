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

#define ERRCHECK() \

#define CHECK_FRAMEBUFFER_STATUS() \
{\
 GLenum status; \
 status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT); \
 switch(status) { \
 case GL_FRAMEBUFFER_COMPLETE_EXT: \
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
/*\
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

/*
 * A callback function for cg to use when it encounters an error
 */
static CGcontext errContext;

static void cgErrorCallbackSum(void) {
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

bool Sum::isValidDimensions( GLuint width, GLuint height ) {
    // verify texture dimensions as valid.
    GLint maxSize; 
    glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxSize );
    if( width > (GLuint)maxSize || height > (GLuint)maxSize ) 
    {
        cerr<<"Sum: Invalid size "<<width<<"x"<<height<<endl;
        assert(0);
        return false;
    }     
    return true;
}

int Sum::numLevels(int W, int H)
{   
    double log2;
    log2 = log(2.0);
    
    double dim = (double)min(W,H); 
    int lvls = (int)(floor(log(dim)/log2));
    return lvls;
}
/**Return the number of reductions this sum object is currently set to do
 * @return the number of reductions this sum object will do.
 */
 ///\brief Get the number of reductions that are performed by the object.
int Sum::nReductions() {
    return N_REDUCE;
}

/* Set the number of reductions this object will perform 
 * @param x The desired number of reductions to perform.
 * @return the number of reductions this object is set to
 */
 //brief Set the number of reductions to perform.
//int Sum::setReductions(int x ) {
//    return (N_REDUCE = x);
//}

int Sum::getBestFold(int dim, int maxFold) {
        assert( maxFold == 2  || maxFold == 4 );

        // try folding by 4 first
        if( dim % 4 == 0 && maxFold == 4 ) {
            return dim/4;
        }
        else if( dim % 2 == 0  ) {
            return dim/2;
        }
        else return dim;
}

// Notes:  8x8 is best for AGP cards
//       16x16 is best for PCI cards
bool Sum::makeStrategy(GLuint w, GLuint h, int numReduce ) {

    sizes.push_back(Coords(w,h));
    while( w > 16 || h > 16 ) {
        unsigned int newW, newH;
        // pick the larger dimension
        if ( w >= h ) { 
            newW = getBestFold(w,2);
            newH = getBestFold(h,2);
        }
        else {
            newW = getBestFold(w,2);
            newH = getBestFold(h,2);
        }
        //make sure dimensions were divisible at least by two
        //assert( newW != w );
        //assert( newH != h );
        if(  newW == w || newH == h ) break;
        w = newW;
        h = newH;
        sizes.push_back(Coords(w,h));
    
        if( numReduce != -1 && sizes.size() > numReduce ) break;
    }

         cerr<<"Strategy for is: "<<endl;
     cerr<<"====="<<endl;
     for( unsigned int i=0 ; i<sizes.size() ; i++ ) {
         cerr<<sizes[i].x()<<"x"<<sizes[i].y()<<endl; 
     }
     cerr<<"====="<<endl;


    return true;
}

/**Creates an object to perform sums of texture data.
 * Internally it always uses a RGBA 32-bit/component floating point
 * summation.  It is also significantly faster to use a 16bit floating point 
 * as input to the summation.  (1.4 to 1.6 times as fast in general).
 * @param width The width of the texture.
 * @param height The height of the texture.
 * @param numReduct (optional) The desired number of reductions to perform 
 *
 */ 
 ///\brief Creates an object to perform sums of texture data.
Sum::Sum(GLuint width, GLuint height, int numReduce ) 
{
    makeStrategy(width,height, -1);
    /* 
     * verify dimensions are valid.
     */ 
    isValidDimensions(width, height);
    this->width = width;
    this->height=height;

    /* 
     *  determine how many times to reduce
     */
    //N_REDUCE = numLevels( (int)width, (int)height);
    N_REDUCE = sizes.size() -1 ;
	
    fb  = (GLuint *)malloc( sizeof( GLuint ) * N_REDUCE );
    tex = (GLuint *)malloc( sizeof( GLuint ) * N_REDUCE );
    /* todo larger than needed for debug */
    rbbuf = (float *)malloc( sizeof(float) * width * height * 4);
    assert( rbbuf!=NULL);
    
    glGenFramebuffersEXT(N_REDUCE, fb);
    glGenTextures(N_REDUCE, tex);       
    errcheck();
    for( int i=0 ; i<N_REDUCE; i++ ) 
    {
        //int lvlW = width/(int)powf(2.0,i+1);
        //int lvlH = height/(int)powf(2.0,i+1);
        int lvlW = (int)sizes[i+1].x();
        int lvlH = (int)sizes[i+1].y();
        /*      cerr<<i<<" : "<<lvlW<<"x"<<lvlH<<endl; */
        //bind the framebuffer, fb, so operations will now occur on it
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb[i]);

        // initialize texture that will store the framebuffer image (first target)
        //glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex[0]);
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex[i]);    
        // sums need max precision/ range. Use FP32 texture
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA32_NV, 
                lvlW, lvlH, 0, GL_RGBA, GL_FLOAT,NULL);
        glTexParameteri(GL_TEXTURE_RECTANGLE_NV,
                        GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
        glTexParameteri(GL_TEXTURE_RECTANGLE_NV,
                        GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_REPLACE );

        //fprintf(stderr,"Generate Mip map ok\n");

        errcheck();

        // bind this texture to the current framebuffer obj. as 
        // color_attachement_0 
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
                GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_NV, tex[i], 0);
        errcheck();

        //see if everything is OK
        CHECK_FRAMEBUFFER_STATUS()
    }
    /*     cerr<<"SUM CREATED"<<endl; */
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);
    init_cg();
    /* Todo: rebind original framebuffer and texture to what it was when we were called */
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0 );
}

/**This returns the sum of the given texture input.
 * 
 * By default only two arguments are needed: the input texture and a
 * pointer to a floating point number.  The floating point number is filled
 * in with the sum of the texture.
 * 
 * Most formats internally will be [0,1.0] per channel.  This function will thus return
 * the sum ranging in [0, W*H*4].
 * 
 * For inputs which hold real floating point Data, such as textures which were
 * GL_FLOAT_RGBA32_NV, this function will return the unclamped sum as well.
 * 
 * This function optionally can produce four sums, one for each channel of the texture.
 * Specifying the last variable (perChannel) as true will produce four sums.  In
 * this case, the result pointer should be a pointer to an array of four floating
 * point numbers.
 * 
 * @param texInput The texture we wish to sum.
 *
 * @param result A pointer to memory where the results will be written to.
 * @param perChannel If true, result should be an array of 4 floats, which is then
 * 					 filled in with the sum of each channel.  If false, the sum is
 * 					 a single floating point value that is the sum of all channels.
 */
///\brief Returns the sum of the texture data.
void Sum::getSum(GLuint texInput,double *result,bool perChannel) 
{
	int previousViewportDims[4] ;
	glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glGetIntegerv(GL_VIEWPORT, previousViewportDims );
    
	int lvl =1;
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glClear(GL_DEPTH_BUFFER_BIT);
    cgGLEnableProfile(cgProfile);
    glActiveTextureARB(GL_TEXTURE0_ARB);
	for( lvl=1 ; lvl<=N_REDUCE; lvl++ ) {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb[lvl-1]);
        if( lvl == 1 ) {
            reduce(texInput, (int)sizes[lvl-1].x(), (int)sizes[lvl-1].y(), 
                    (int)sizes[lvl].x(), (int)sizes[lvl].y() );
        }
        else {
            reduce(tex[lvl-2], (int)sizes[lvl-1].x(), (int)sizes[lvl-1].y(), 
                    (int)sizes[lvl].x(), (int)sizes[lvl].y() );
        }
        //glReadPixels( 0, 0,width/2,height/2, GL_RGBA, GL_FLOAT, rbbuf);
        //cerr<<sumBuf(rbbuf,width/32,height/32,4);
    }

    int rbW = (int)sizes[lvl-1].x();
    int rbH = (int)sizes[lvl-1].y();
//    cerr<<"Reading back "<<rbW<<"x"<<rbH<<endl;
    
    glReadPixels( 0, 0,rbW, rbH, GL_RGBA, GL_FLOAT, rbbuf);
    
    if( perChannel ) {
    	sumBuf4( rbbuf,rbW, rbH,4, &(result[0]), &(result[1]), &(result[2]), &(result[3]) );
    }
    else {
    	*result = sumBuf(rbbuf,rbW, rbH,4);
    }
    /* todo : rebind original callingframebuffer and texture state */
    /* todo : push pop matrices */
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0 );
    glActiveTextureARB(GL_TEXTURE0_ARB);
 
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glViewport( previousViewportDims[0], previousViewportDims[1],
    previousViewportDims[2], previousViewportDims[3] );
    cgGLDisableProfile(cgProfile);
    return;
}



/**Destroy the Sum object.
 */
 ///\brief Destroy the Sum object.
Sum::~Sum()
{
    glDeleteTextures( N_REDUCE, tex );
    glDeleteFramebuffersEXT( N_REDUCE, fb );
    free(fb);
    free(tex);
    free(rbbuf);
}

void Sum::reduce(GLuint texture, int w, int h, int W, int H) {
    reshape( W, H );

    float eps = 0.5;
    float tap4X, tap4Y, tap5X, tap5Y, tap6X, tap6Y, tap7X, tap7Y;


    if( w/W == 2 && h/H == 2 ) {
        cgGLBindProgram(fragmentProgram4tap);
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, texture );
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_REPLACE );
        glBegin(GL_QUADS);
        glMultiTexCoord2f( GL_TEXTURE0_ARB,  0.0+eps  , 0.0+eps  );
        glMultiTexCoord2f( GL_TEXTURE1_ARB,  0.0-1+eps  , 0.0 +eps );
        glMultiTexCoord2f( GL_TEXTURE2_ARB,  0.0 +eps , -1.0 +eps );
        glMultiTexCoord2f( GL_TEXTURE3_ARB,  0.0-1 +eps , -1.0 +eps );
        glVertex2f( 0, 0);

        glMultiTexCoord2f( GL_TEXTURE0_ARB, (float)w +eps , 0.0+eps );
        glMultiTexCoord2f( GL_TEXTURE1_ARB, (float)w-1 +eps , 0.0+eps );
        glMultiTexCoord2f( GL_TEXTURE2_ARB, (float)w +eps , -1.0+eps );
        glMultiTexCoord2f( GL_TEXTURE3_ARB, (float)w-1 +eps , -1.0+eps );
        glVertex2f( W,   0);

        glMultiTexCoord2f( GL_TEXTURE0_ARB, (float)w+eps , (float)h+eps );
        glMultiTexCoord2f( GL_TEXTURE1_ARB, (float)w-1+eps , (float)h+eps );
        glMultiTexCoord2f( GL_TEXTURE2_ARB, (float)w+eps , (float)h-1.0+eps );
        glMultiTexCoord2f( GL_TEXTURE3_ARB, (float)w-1+eps , (float)h-1.0+eps );

        glVertex2f(W,H);

        glMultiTexCoord2f( GL_TEXTURE0_ARB, 0.0+eps, (float)h+eps  );
        glMultiTexCoord2f( GL_TEXTURE1_ARB, 0.0-1+eps, (float)h+eps  );
        glMultiTexCoord2f( GL_TEXTURE2_ARB, 0.0+eps, (float)h-1.0+eps  );
        glMultiTexCoord2f( GL_TEXTURE3_ARB, 0.0-1+eps, (float)h-1.0 +eps );
        glVertex2f( 0, H);

        glEnd(); 
        errcheck();
        return;
    }

    else if ( w/W == 4 && h/H == 2) {

    tap4X = -2.0 + eps;
    tap4Y = -1.0 + eps;

    tap5X = +1.0 + eps;
    tap5Y = -1.0 + eps;

    tap6X = -2.0 + eps;
    tap6Y =  0.0 + eps;
    
    tap7X = +1.0 + eps;
    tap7Y =  0.0 + eps;
    cgGLBindProgram(fragmentProgram8tap);

    } else if ( w/W == 2 && h/H == 4 ) {

    tap4X = -1.0 + eps;
    tap4Y = +1.0 + eps;

    tap5X =  0.0 + eps;
    tap5Y = +1.0 + eps;

    tap6X = -1.0 + eps;
    tap6Y = -2.0 + eps;
    
    tap7X =  0.0 + eps;
    tap7Y = -2.0 + eps;
    cgGLBindProgram(fragmentProgram8tapY);

    } else {
        assert(0);
    }
    //cerr<<"Red from "<<w<<"x"<<h<<" to "<<W<<"x"<<H<<endl;
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, texture );
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_REPLACE );

    glBegin(GL_QUADS);
    glMultiTexCoord2f( GL_TEXTURE0_ARB,  0.0+eps  , 0.0+eps  );
    glMultiTexCoord2f( GL_TEXTURE1_ARB,  0.0-1+eps  , 0.0 +eps );
    glMultiTexCoord2f( GL_TEXTURE2_ARB,  0.0 +eps , -1.0 +eps );
    glMultiTexCoord2f( GL_TEXTURE3_ARB,  0.0-1 +eps , -1.0 +eps );

    glMultiTexCoord2f( GL_TEXTURE4_ARB,  tap4X, tap4Y);
    glMultiTexCoord2f( GL_TEXTURE5_ARB,  tap5X, tap5Y );
    glMultiTexCoord2f( GL_TEXTURE6_ARB,  tap6X, tap6Y );
    glMultiTexCoord2f( GL_TEXTURE7_ARB,  tap7X, tap7Y );
    glVertex2f( 0, 0);

    glMultiTexCoord2f( GL_TEXTURE0_ARB, (float)w +eps , 0.0+eps );
    glMultiTexCoord2f( GL_TEXTURE1_ARB, (float)w-1 +eps , 0.0+eps );
    glMultiTexCoord2f( GL_TEXTURE2_ARB, (float)w +eps , -1.0+eps );
    glMultiTexCoord2f( GL_TEXTURE3_ARB, (float)w-1 +eps , -1.0+eps );

    glMultiTexCoord2f( GL_TEXTURE4_ARB, (float)w +tap4X , tap4Y);
    glMultiTexCoord2f( GL_TEXTURE5_ARB, (float)w +tap5X , tap5Y);
    glMultiTexCoord2f( GL_TEXTURE6_ARB, (float)w +tap6X , tap6Y);
    glMultiTexCoord2f( GL_TEXTURE7_ARB, (float)w +tap7X , tap7Y);
    glVertex2f( W,   0);

    glMultiTexCoord2f( GL_TEXTURE0_ARB, (float)w+eps , (float)h+eps );
    glMultiTexCoord2f( GL_TEXTURE1_ARB, (float)w-1+eps , (float)h+eps );
    glMultiTexCoord2f( GL_TEXTURE2_ARB, (float)w+eps , (float)h-1.0+eps );
    glMultiTexCoord2f( GL_TEXTURE3_ARB, (float)w-1+eps , (float)h-1.0+eps );

    glMultiTexCoord2f( GL_TEXTURE4_ARB, (float)w+tap4X, (float)h+tap4Y );
    glMultiTexCoord2f( GL_TEXTURE5_ARB, (float)w+tap5X, (float)h+tap5Y );
    glMultiTexCoord2f( GL_TEXTURE6_ARB, (float)w+tap6X, (float)h+tap6Y );
    glMultiTexCoord2f( GL_TEXTURE7_ARB, (float)w+tap7X, (float)h+tap7Y );
    glVertex2f(W,H);

    glMultiTexCoord2f( GL_TEXTURE0_ARB, 0.0+eps, (float)h+eps  );
    glMultiTexCoord2f( GL_TEXTURE1_ARB, 0.0-1+eps, (float)h+eps  );
    glMultiTexCoord2f( GL_TEXTURE2_ARB, 0.0+eps, (float)h-1.0+eps  );
    glMultiTexCoord2f( GL_TEXTURE3_ARB, 0.0-1+eps, (float)h-1.0 +eps );

    glMultiTexCoord2f( GL_TEXTURE4_ARB, 0.0+tap4X, (float)h+tap4Y  );
    glMultiTexCoord2f( GL_TEXTURE5_ARB, 0.0+tap5X, (float)h+tap5Y  );
    glMultiTexCoord2f( GL_TEXTURE6_ARB, 0.0+tap6X, (float)h+tap6Y  );
    glMultiTexCoord2f( GL_TEXTURE7_ARB, 0.0+tap7X, (float)h+tap7Y  );
    glVertex2f( 0, H);

    glEnd(); 


/*
  float r[4];
    glReadPixels(0,0,1,1,GL_RGBA,GL_FLOAT,r);
    cerr<<"r=["<<r[0]<<","<<r[1]<<","<<r[2]<<","<<r[3]<<"]"<<endl;
*/



 

   errcheck();
}


void Sum::errcheck() {
static GLenum errCode;
const GLubyte *errString;
  if ((errCode = glGetError()) != GL_NO_ERROR) {
    errString = gluErrorString(errCode);
    fprintf (stderr, "OpenGL Error: %s\n", errString);
    exit(1);
  }

}

void Sum::reshape(int w, int h)
{
   // viewport for 1:1 pixel=texel mapping
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, w, 0.0, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0,w, h);
}

void Sum::init_cg() {
  cgProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT) ;
  cgContext = cgCreateContext();
  errContext = cgContext;
  cgSetErrorCallback(cgErrorCallbackSum);
  
  
  fragmentProgram4tap = cgCreateProgramFromFile( cgContext, 
                                             CG_SOURCE, "FP-4tap.cg", 
                                             cgProfile, "FragmentProgram", 0);
 
  fragmentProgram8tap= cgCreateProgramFromFile( cgContext, 
                                             CG_SOURCE, "FP-8tap.cg", 
                                             cgProfile, "FragmentProgram", 0);
  fragmentProgram8tapY= cgCreateProgramFromFile( cgContext, 
                                             CG_SOURCE, "FP-8tapY.cg", 
                                             cgProfile, "FragmentProgram", 0);
  cgGLLoadProgram( fragmentProgram4tap );
  cgGLLoadProgram( fragmentProgram8tap );
  cgGLLoadProgram( fragmentProgram8tapY);
}

double Sum::sumBuf( float *b, int w, int h, int c) {
    int i;
    double tmp=0.0;
    for( i=0 ; i<w*h*c ; i++ ) { tmp+=(double)b[i]; }
    return tmp;
}

void Sum::sumBuf4( float *b, int w, int h, int c, double *r1, double *r2, double *r3, double *r4) 
{
    int i;
    *r1 = 0.0;
    *r2 = 0.0;
    *r3 = 0.0;
    *r4 = 0.0;
    for( i=0 ; i<w*h*c ; i+=4 ) { 
    	*r1+=(double)b[i]; 
    	*r2+=(double)b[i+1]; 
    	*r3+=(double)b[i+2]; 
    	*r4+=(double)b[i+3]; 
    }
   //cerr<<"The 4 buffer sum is "<<*r1+*r2+*r3+*r4<<endl;
    //cerr<<"The 1 element sum is"<<sumBuf(b,w,h,c);
    return;
}

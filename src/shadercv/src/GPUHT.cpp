#include <GL/gl.h>
#include <GL/glut.h>
#include <Cg/cgGL.h>
#include <stdio.h>
#include <cmath>
#include <assert.h>
#include <iostream>

#define PI 3.141592653589793116

using namespace std;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

/*
 * A callback function for cg to use when it encounters an error
 */
static CGcontext errContextGPUHT;

static void cgErrorCallbackGPUHT(void) {
    CGerror LastError = cgGetError();

    if(LastError)
    {
        const char *Listing = cgGetLastListing(errContextGPUHT);
        printf("\n---------------------------------------------------\n");
        printf("%s\n\n", cgGetErrorString(LastError));
        printf("%s\n", Listing);
        printf("---------------------------------------------------\n");
        printf("Cg error, exiting...\n");
        exit(0);
    }
}

class houghLine : public pair<float,float>
{
  public:
    float theta() { return first; }
    float r() { return second; }
    pair<float,float> pointA; 
    pair<float,float> pointB; 

    houghLine( float inTheta, float inR ) 
    {
        float A, B, C;
        first = inTheta;
        second = inR;
        pointA.first =0.0;
        pointA.second =0.0;
        pointB.first =0.0;
        pointB.second =0.0;

        if( theta() >= 0.25*PI && theta() <= 0.75*PI ) {
            A = cosf( theta() );     
            B = sinf( theta() );     
        
            pointA.first = 0.0;
            pointA.second =  -r()/B;

        // why 1.3?
        // choose a X axis point far enough away to span the image.
        // for a 4/3 image, this point is 4/3 away ~= 1.333 < 1.4
            pointB.first = 1.4;
            pointB.second = ( -A/B*1.4 - r()/B );
        }
        else {

             A = cosf( theta() );     
             B = sinf( theta() );     

            pointA.first = -r()/A;
            pointA.second = 0.0;
            pointB.first = -B/A - r()/A ;
            pointB.second = 1.0 ;

        }
    }
    void draw() {
        //cerr<<"h["<<theta()<<","<<r()<<"]  "; 
        glVertex3f( pointA.first, pointA.second, -1.0 );
        glVertex3f( pointB.first, pointB.second, -1.0 );
        //cerr<<"[ "<<pointA.first<<", "<<pointA.second<<"] --> [";
        //cerr<<pointB.first<<", "<<pointB.second<<"]\n";
    }
};

class GPUHT {

    int method;

    //CFPBuffer fpbuffer ;
    GLuint vbo;

    // resWidth and resHeight are width/height of the hough space. 
    int resWidth, resHeight;

    // a buffer to hold edgels. 
    float *edgels; 
  

    // projection matrix
    double ProjectionMatrix[16];

    CGprogram basicProgram, VisProgram;
    CGprofile cgVtxProfile, cgFrgProfile;
    CGcontext cgContext; 

    //Given an input at datatpr, look for edgels by testing the red channel.
    //TODO : change the function to work with less than 4 channels, since it
    //      is inly interested in 1 channel anyways.
    int findEdgels(unsigned char *dataptr, float *ptr, int W, int H, int nchan,
                   unsigned char thresh ) 
    {
       int numEdgels = 0;

       for( int i=0 ; i<H ; i++ ) {
            for( int j=0 ; j<W ; j++ ) {
                if( *dataptr  >= thresh ) { // if edgel


                    if( usePoints() ) { 
                      //*ptr++ = ((float)j+0.5)/(float)H;
                      //*ptr++ = ((float)i+.5)/(float)H;
                        // no offset is more accurate on points
                        // agreeing with ceiling rounding  in octave
                      *ptr++ = ((float)j)/(float)H;
                      *ptr++ = ((float)i)/(float)H;
                      *ptr++ = 0.0;
                      *ptr++ = 1.0;
                    } else {
                        //direction makes no difference
                     // *ptr++ = ((float)j+0.5)/(float)H;
                     // *ptr++ = ((float)i+.5)/(float)H;
                      *ptr++ = ((float)j)/(float)H;
                      *ptr++ = ((float)i)/(float)H;
                      *ptr++ = 1.0;
                      *ptr++ = 0.0;
                      //*ptr++ = ((float)j+.5)/(float)H;
                      //*ptr++ = ((float)i+.5)/(float)H;
                      *ptr++ = ((float)j)/(float)H;
                      *ptr++ = ((float)i)/(float)H;
                      *ptr++ = 0.0;
                      *ptr++ = 1.0;
                    }
                    numEdgels++;
                    //cerr<<"Marked: "<< (float)j/(float)H <<","<<(float)i/(float)H<<endl;

                }
                dataptr+=nchan;
            }
       }
    //   cerr<<"Found "<<numEdgels<<" edgels\n"<<endl;
       return numEdgels ;
    }

    void newMat(  double theta , double NextTheta)
    {
        ProjectionMatrix[1]  = cos( theta*PI ) ;
        ProjectionMatrix[5]  = sin( theta*PI ) ;

        //normaliz  for ndcs
        ProjectionMatrix[12] =  (theta) ;
if( !usePoints() ) {
        ProjectionMatrix[8]  =  ((NextTheta))  ;
        ProjectionMatrix[2]  = (cos(NextTheta*PI) );
        ProjectionMatrix[6]  = (sin(NextTheta*PI) );
}

        glMatrixMode(GL_PROJECTION);
        glLoadMatrixd( ProjectionMatrix);

    }

    CGprogram load_cgprogram(CGprofile prof, char *name) {
        fprintf(stderr, "loading %s\n", name);
        return cgCreateProgramFromFile( cgContext, CG_SOURCE,
                                  name, prof, "FragmentProgram", 0);
    }

   
    public: 
        ~GPUHT() { free( edgels ); }
        bool usePoints() { return true; }

        GPUHT( int resolutionW, ///< Width of transform.  Affects the resolution of the angle theta of transform.
                int resolutionH,  ///< Height of transform.  Affects the resoution of the distance of perpendicular to line.
                int imgW, 
                int imgH ) {

            if( glutGet(GLUT_WINDOW_STENCIL_SIZE) != 8  ) {
               cerr<<"[Hough] **ERROR** An 8 bit stencil is needed.  Try glutInitDisplay(..|GLUT_STENCIL...)"<<endl;
               exit(1);
            }


            resWidth = resolutionW;
            resHeight = resolutionH;

            edgels = (float *)malloc( sizeof(float) * imgW * imgH * 4 );
            //fpbuffer.create(resolutionW, resolutionH ); 
            //fpbuffer.activate();
            //glViewport(0,0,(GLsizei)resolutionW, (GLsizei) resolutionH);


            //create a vertex buffer object to hold edgel data since we will
            // be drawing it multiple times, hopefully this will maximize
            // efficiency of the draw ops.

            // Leaves the door open for streaming vertex arrays using
            // Map/Unmap.

            //This also leaves the door open for a render-to-vertex-array
            // functionality.  However, in tests, for 320x240, it takes about
            // 76fps (13ms) (excluding vertex program) to process 
            // 320x240x2 line vertices, and since most of the vertices are
            // non edgels anyways its more efficient to do a readback
            // pack the vertices and send back only the edgel data.
            // especially with PCIe around. 

            //note: an IF statement on the vertex prog drops vertex rate to
            // 30 FPS from 76 FPS.  The "masking" op on the lines 
            // goes at 64 FPS (timings on Go 6600, 320x240x2 GL_LINES vert)
            
            glGenBuffersARB( 1, &vbo );
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
            glBufferDataARB(GL_ARRAY_BUFFER_ARB,  
//                            sizeof(float)*4*2*imgW*imgH  ,
                            sizeof(float)*4*2*imgW*imgH/4  ,
                            NULL, GL_STATIC_DRAW_ARB);
            glVertexPointer( 4, GL_FLOAT, 0, BUFFER_OFFSET(0) );

            cgVtxProfile = CG_PROFILE_VP20;
            cgFrgProfile = CG_PROFILE_FP30;

            cgContext = cgCreateContext(); 
            errContextGPUHT = cgContext; 
            cgSetErrorCallback(cgErrorCallbackGPUHT);
            
#define CGDIR SRCDIR"/src/shadercv/shaders"
            if( usePoints() ) { 
                basicProgram     = load_cgprogram(cgVtxProfile, CGDIR"/VP-points.cg");
            } else {
                basicProgram     = load_cgprogram(cgVtxProfile, CGDIR"/VP-basic.cg");
            }
            cgGLLoadProgram( basicProgram );
            VisProgram     = load_cgprogram(cgFrgProfile, CGDIR"/FP-houghVis.cg");
            cgGLLoadProgram( VisProgram );

            //fpbuffer.deactivate();

            for( int i=0; i<16; i++ ) ProjectionMatrix[i] = 0.0;
        }

        int houghWidth() { return resWidth; }  ///< Returns width of transform.
        int houghHeight() { return resHeight; }  ///<Returns height of transform.

        int getVotes(unsigned char *votesBuf,   ///< Target buffer to be filled in
                                                ///  with votes.  Allocated by
                                                /// the caller, and fille in by
                                                /// this function.  The buffer
                                                /// 8 bits per pixel, and
                                                /// of dimensions equivalent to 
                                                /// the resolutionW/resolutionH
                                                /// of the transform.
                      unsigned char *edgelsImg, ///< Source buffer holding the
                                                /// edgels to transform.
                      int W,   ///< Width of edgelsImg
                      int H,   ///< Height of edgelsImg 
                      int steps   )  ///<number of steps to interpolate.  Affercts
                                     /// the resolution of the approximation of the
                                     /// transform to hough space curves.  Typically
                                     /// set steps equal to the hough ResolutionWidth.  
        { 
            CGparameter modelViewMatrix; 
            // fill in an array of line vertices at edgel locations.
            unsigned char thresh = 200;
            int numE = findEdgels( edgelsImg, edgels, W, H, 3, thresh );

            //fpbuffer.activate(); //XXX adds +3 msecs switching context

            int previousViewportDims[4];
            // XXX adds 0.3 msecs getting state.
            glGetIntegerv(GL_VIEWPORT, previousViewportDims );
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();


            glViewport(0, 0, (GLsizei)resWidth, (GLsizei) resHeight);

            //stencil accumulation is faster than additive blend.
            glClear(GL_STENCIL_BUFFER_BIT);
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

            // update the VBO, push the data up using subData.
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo );


            //each edgel generates 2 vertices, line endpoints

            glBufferSubDataARB( GL_ARRAY_BUFFER_ARB,  0,
                                numE*sizeof(float)*4*2 , 
                                edgels );


            glEnableClientState( GL_VERTEX_ARRAY );
            cgGLEnableProfile(cgVtxProfile);
            cgGLBindProgram(basicProgram);

            modelViewMatrix = cgGetNamedParameter(basicProgram, 
                                                 "ModelViewProj");
            cgGLSetStateMatrixParameter(modelViewMatrix, 
                                CG_GL_MODELVIEW_PROJECTION_MATRIX,
                                CG_GL_MATRIX_IDENTITY);
            glEnable(GL_STENCIL_TEST);
            glStencilFunc(GL_NEVER, 0x0, 0x0);
            glStencilOp(GL_INCR, GL_INCR, GL_INCR);

            for(int i=0; i<steps; i+=1 ) {
                //newMat( (float)(i)*PI/(float)steps,
                //    (float)(i+1.0)*PI/(float)steps );
                newMat( ((double)(i))/(double)steps,
                    (double)(i+1)/(double)steps );
                cgGLSetStateMatrixParameter(modelViewMatrix,
                             CG_GL_MODELVIEW_PROJECTION_MATRIX,
                             CG_GL_MATRIX_IDENTITY);
                //glLineWidth(0.5);
                if( usePoints() ) {
                    glDrawArrays(GL_POINTS, 0 , numE ); 
                } else {
                    glDrawArrays(GL_LINES, 0, numE*2 );
                }
            }
            glDisableClientState( GL_VERTEX_ARRAY );
            cgGLDisableProfile(cgVtxProfile);

            glDisable(GL_STENCIL_TEST);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

            // Read back the votes buf into the supplied pointer area.
            glReadPixels(0,0,resWidth,resHeight,GL_STENCIL_INDEX, 
                    GL_UNSIGNED_BYTE, votesBuf);

            //fpbuffer.deactivate();  
            glViewport( previousViewportDims[0], previousViewportDims[1],
                  previousViewportDims[2], previousViewportDims[3] );
			glMatrixMode(GL_MODELVIEW);
            glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            
            return numE ;
        }

};

/* COMPILE: 
   g++ cgExample.cc  -I../include -L../ -I/usr/include/cc++ -I/usr/include/cc++2/ -ldc1394_control -lraw1394  -lstdc++ -lccext2 -lccgnu2 -lxml -lopenvidia -lpthread -lGL -lglut -DGL_GLEXT_PROTOTYPES -DGLX_GLXEXT_PROTOTYPES -lImlib -I~/SDK/LIBS/inc -lCgGL

*/

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <Imlib.h>
#include <openvidia/openvidia32.h>

#include "../src/shadercv/src/GPUHT.cpp"

#define CGDIR SRCDIR"/src/shadercv/shaders"

using namespace std;
GPUHT * hough;
GLuint houghtex, occlusionQueries;
unsigned char *cannyEdgels, *votesBuf;

CGprogram basicProgram, visProgram;
CGprofile cgProfile;
CGcontext cgContext;
int hw = 320;
int hh = 320;
int HOUGH_THRESH=80;
int width,height;


vector<Coords> taps;
ImlibImage *Im;
GLuint tex, oTex ;  
Timer t;
FileFrame *ff;
FBO_Filter *filtUndistort, *filtDerivs, *filtCanny ;

vector<houghLine> HoughLines;
/**
 * 320x240- ff-width()/2 ~= 160
 * 640x480- 200;
 * 
 */
//f - Radon Transform
//W, H - width and height of the transform 
void analyze( unsigned char *f, int W, int H  ) {

	HoughLines.clear(); 

 	int numPoints = 0, pos = 0;
	for(int i=0; i<H ; i++ ) {
 	   	int rowoffset = i*W;
 	   	for(int j=0; j<W ; j++ ) {
			if( f[ (rowoffset+j) ]  > HOUGH_THRESH ) {
 	      	//cerr<<"j,i is "<<j<<","<<i<<" "<<endl;
 	      	houghLine hl = houghLine( (float)j/(float)W*3.14159, 
                                 //(float)(abs(i-H/2))/(float)(H/2) *1.414);
                                 (float)(i-H/2)/(float)(H/2) *1.414);
       		HoughLines.push_back( hl );
       		hl.draw();
      		}
    	}
  	}
  	endit:
  	//  cerr<<"Found "<<HoughLines.size()<<" houghlines.\n";
  	return;
}


CGprogram load_cgprogram(CGprofile prof, std::string name) {
  fprintf(stderr, "loading %s\n", name.c_str ());
  return cgCreateProgramFromFile( cgContext, CG_SOURCE,
                                  name.c_str (), prof, "FragmentProgram", 0);
}

void reshape(int w, int h)
{
	w=w*2;
	h=h*2;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, w, 0,h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, w, h);
}

void myIdle()
{
    glutPostRedisplay();
}

void keyboard (unsigned char key, int x, int y)
{
    switch (key) {
        case 27:
            exit(0);
            break;
        case '+':
            HOUGH_THRESH += 10;
            break;
        case '-':
            HOUGH_THRESH -= 10;
            break;
        default:
            break;
    }
}

void MouseFunc( int button, int state, int x, int y)
{
    switch(button) 
    {
        case GLUT_LEFT_BUTTON :
            break;
        case GLUT_RIGHT_BUTTON :
            break;
    }
}

int c=0;
void render_redirect() 
{
    float d =  -1.0;

    glClearColor(0.0, 0.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, filtCanny->tex() ) ;
    //ff->tex() ;
    // draw the 'tex' texture containing the framebuffer rendered drawing.
    glBegin(GL_QUADS); 
    
		glTexCoord2f(0, ff->height());
		glVertex3f( 0.0 , ff->height() , d );
		
		glTexCoord2f(0, 0);
		glVertex3f( 0.0, 0.0, d );
		
		glTexCoord2f(ff->width(), 0);
		glVertex3f( ff->width(), 0.0, d );
		
		glTexCoord2f(ff->width(),ff->height() );
		glVertex3f( ff->width(), ff->height(), d );
	glEnd();
	
	
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, filtUndistort->tex_nodirty() ) ;

    // draw the 'tex' texture containing the framebuffer rendered drawing.
    glBegin(GL_QUADS); 
		glTexCoord2f(0, ff->height());
		glVertex3f( 0.0, ff->height() + ff->height(), d );
		
		glTexCoord2f(0, 0);
		glVertex3f( 0.0, 0.0+ ff->height(), d );
		
		glTexCoord2f(ff->width(), 0);
		glVertex3f( ff->width(), 0.0+ ff->height(), d );
		
		glTexCoord2f(ff->width(),ff->height() );
		glVertex3f( ff->width(), ff->height()+ ff->height(), d );
	glEnd();

	glReadPixels( 0, 0, ff->width(), ff->height(), GL_RGB, GL_UNSIGNED_BYTE, cannyEdgels);
/*
    for( int i=0 ; i<320*240*3; i++ ) {
       cannyEdgels[i] = 255;
     }

    int pos = 320/2 * 240/2 * 3;
    cannyEdgels[pos] = 255;
*/


	int edgels = hough->getVotes( votesBuf, cannyEdgels, ff->width(), ff->height(), 24);
	
	glActiveTextureARB(GL_TEXTURE0_ARB);
 	glBindTexture(GL_TEXTURE_RECTANGLE_NV, houghtex) ;
 	glTexSubImage2D( GL_TEXTURE_RECTANGLE_NV, 0, 0,0, hw, hh,
                   GL_RED, GL_UNSIGNED_BYTE, votesBuf );

  	cgGLEnableProfile(cgProfile);
  	cgGLBindProgram(visProgram);
  	glBegin(GL_QUADS);

    glTexCoord2f(0, hh);
    glVertex3f(ff->width(), 0.0+ ff->height(),d );

    glTexCoord2f(0, 0);
    glVertex3f(ff->width(), hh + ff->height(), d);

    glTexCoord2f(hw, 0);
    glVertex3f(ff->width()+hw,hh+ ff->height(), d);

    glTexCoord2f(hw,hh);
    glVertex3f(ff->width()+hw, 0.0+ ff->height(), d );
  	glEnd();
	
	cgGLDisableProfile(cgProfile);

	// draw the undistorted texture again, the lines will be overlayed on top.
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, filtUndistort->tex_nodirty() ) ;
    glBegin(GL_QUADS); 
		glTexCoord2f(0, ff->height());
		glVertex3f( width, ff->height() , d );
		glTexCoord2f(0, 0);
		glVertex3f( width, 0, d );
		glTexCoord2f(ff->width(), 0);
		glVertex3f( ff->width()*2,0, d );
		glTexCoord2f(ff->width(),ff->height() );
		glVertex3f( ff->width()*2, ff->height(), d );
	glEnd();
	
	// disable the textureing. we are going to draw lines now
	glDisable(GL_TEXTURE_RECTANGLE_NV);
	// reshape the viewport to focus on the corner with the image where we
	// will overlay the lines
	glViewport(width,0,width,height);
	// reset the viewing parameters for our line drawing.
  	glMatrixMode(GL_PROJECTION);
  	glLoadIdentity();
  	glFrustum(0.0, 4.0/3.0,  0.0, 1.0,   1.0,   100.0);
  	gluLookAt(0.0,0.0,0.0,  0.0, 0.0,  -1.0,   0.0, 1.0, 0.0);
  	//emit the line segments
 	glBegin(GL_LINES);
   		analyze( votesBuf, hw, hh );
 	glEnd(); 
	reshape(width,height);

	glutSwapBuffers();
}

extern "C" void glewInit ();
int main(int argc, char *argv[] )  
{/*
    if( argc != 3 ) {
        fprintf(stderr, "supply 2 parameters:and a cg program filename, and an image filename.\n");
        exit(0);
    }
*/
	
    Coords tap1(-2,0);
    Coords tap2(-1,0);
    Coords tap3( 0,0);
    Coords tap4(+1,0);
    Coords tap5(+2,0);

    taps.push_back(tap1);
    taps.push_back(tap2);
    taps.push_back(tap3);
    taps.push_back(tap4);
    taps.push_back(tap5);
    
    
	glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL );
    glutInitWindowSize( 320,240 );
	glutCreateWindow(argv[0]);
	
	ff = new FileFrame(argv[1]);
	width = ff->width();
	height = ff->height();
	//glutReshapeWindow( ff->width()*2, ff->height()*2 );
	glutReshapeWindow( ff->width()*4, ff->height()*4 );
	reshape( ff->width()*2, ff->height()*2 );
    filtUndistort = new FBO_Filter( CGDIR"/FPcanny/FP-func-undistort.cg", 
    								ff, GL_RGBA, NULL , NULL);
    filtDerivs = new FBO_Filter( CGDIR"/FPcanny/FPderivs.cg",
    							filtUndistort, GL_FLOAT_RGBA32_NV, NULL, NULL );
    filtCanny = new FBO_Filter( CGDIR"/FPcanny/FP-canny-search.cg",
    							filtDerivs, GL_RGBA, NULL, NULL );
    
	cerr<<"Size = "<<ff->width()<<"x"<<ff->height() << endl;
    cout<<"Vendor : " <<glGetString(GL_VENDOR)<<endl;
    cout<<"Renderer : " <<glGetString(GL_RENDERER)<<endl;
    cout<<"Version : " <<glGetString(GL_VERSION)<<endl;
  	glGenTextures(1 , &houghtex ); 
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, houghtex);
 	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA, hw,hh, 0,
                GL_RGB, GL_UNSIGNED_BYTE,NULL );

 	hough = new GPUHT( hw,hh,ff->width(),ff->height() );
 	cannyEdgels = (unsigned char *)malloc(3*ff->width()*ff->height());
	votesBuf = (unsigned char *)malloc(hw*hh);

	//in order to see some of the processing buffers, we need 
	// a fragment program because the floating point values wont 
	// show up on their own.  FP-basic.cg just pass texels through.
	cgProfile = CG_PROFILE_FP40;
	cgContext = cgCreateContext();
	basicProgram     = load_cgprogram(CG_PROFILE_FP40, CGDIR"/FP-basic.cg");
	visProgram     = load_cgprogram(CG_PROFILE_FP40, CGDIR"/FP-houghVis.cg");
	cgGLLoadProgram( basicProgram );
	cgGLLoadProgram( visProgram );

    glutDisplayFunc(render_redirect);
    glutIdleFunc(myIdle);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(MouseFunc);
    glutMainLoop();
    return 0;
}

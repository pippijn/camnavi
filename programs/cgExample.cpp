/* COMPILE: 
   g++ cgExample.cc  -I../include -L../ -I/usr/include/cc++ -I/usr/include/cc++2/ -ldc1394_control -lraw1394  -lstdc++ -lccext2 -lccgnu2 -lxml -lopenvidia -lpthread -lGL -lglut -DGL_GLEXT_PROTOTYPES -DGLX_GLXEXT_PROTOTYPES -lImlib -I~/SDK/LIBS/inc -lCgGL

*/

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <openvidia/openvidia32.h>

using namespace std;
vector<Coords> taps;
GLuint tex, oTex ;  
Timer t;
FileFrame *ff;
FBO_Filter *filter ;

void reshape(int w, int h)
{
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

int sleeptime = 0;

void keyboard (unsigned char key, int x, int y)
{
    switch (key) {
        case '+' : 
            sleeptime += 500;
            break;
        case '-' : 
            sleeptime -= 500;
            break;
        case 27:
            exit(0);
            break;
        default:
            break;
    }
            cerr<<"stime : "<<sleeptime<<endl;
}

void showstats() {
  static int lasttime;
  static int fpscounter; 
  static int fps;
  int curtime;
  curtime = time(NULL);
  if( lasttime != curtime) {
    fps=fpscounter;
    fpscounter=1;
    lasttime = curtime;
    fprintf(stderr, "fps = %d,  %f msecs\n",fps,1.0/((float)fps)*1000);
  } else {
    fpscounter++;
  }
}


void MouseFunc( int button, int state, int x, int y)
{
    switch(button) 
    {
        case GLUT_LEFT_BUTTON :
//            sleeptime += 500;
            break;
        case GLUT_RIGHT_BUTTON :
            //sleeptime -= 500;
            break;
    }
            //cerr<<"stime : "<<sleeptime<<endl;
}


int c=0;
void render_redirect() 
{
    float d =  -1.0;
    filter->resetDirty();
    assert(filter->dirty());
    //glClearColor(0.0, 0.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, filter->tex() ) ;

    // draw the 'tex' texture containing the framebuffer rendered drawing.
    glBegin(GL_QUADS); 
		glTexCoord2f(0, ff->height());
		glVertex3f( 0.0, ff->height(), d );
		
		glTexCoord2f(0, 0);
		glVertex3f( 0.0, 0.0, d );
		
		glTexCoord2f(ff->width(), 0);
		glVertex3f( ff->width(), 0.0, d );
		
		glTexCoord2f(ff->width(),ff->height() );
		glVertex3f( ff->width(), ff->height(), d );
	glEnd();
	glDisable(GL_TEXTURE_RECTANGLE_NV);
      //struct timeval tv; tv.tv_sec = 0; tv.tv_usec = sleeptime;
      //select(0,0,0,0, &tv);

	glutSwapBuffers();

    showstats();
}

int main(int argc, char *argv[] )  
{
    if( argc != 3 ) {
        fprintf(stderr, "supply 2 parameters:and a cg program filename, and an image filename.\n");
        exit(0);
    }

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
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize( 256, 256 );
	glutCreateWindow(argv[0]);
	
	ff = new FileFrame(argv[2]);
	glutReshapeWindow( ff->width(), ff->height() );
    filter = new FBO_Filter( argv[1], ff, GL_RGBA, &taps, NULL);
    
	cerr<<"Size = "<<ff->width()<<"x"<<ff->height() << endl;
    cout<<"Vendor : " <<glGetString(GL_VENDOR)<<endl;
    cout<<"Renderer : " <<glGetString(GL_RENDERER)<<endl;
    cout<<"Version : " <<glGetString(GL_VERSION)<<endl;

    glutDisplayFunc(render_redirect);
    glutIdleFunc(myIdle);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(MouseFunc);
    glutMainLoop();
    return 0;
}



/**** Menu Handling ***/

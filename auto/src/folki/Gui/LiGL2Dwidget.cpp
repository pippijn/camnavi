/*
      This file is part of FolkiGpu.

    FolkiGpu is free software: you can redistribute it and/or modify
    it under the terms of the GNU Leeser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Leeser General Public License
    along with FolkiGpu.  If not, see <http://www.gnu.org/licenses/>.

*/

/*
      FolkiGpu is a demonstration software developed by Aurelien Plyer during
    his phd at Onera (2008-2011). For more information please visit :
      - http://www.onera.fr/dtim-en/gpu-for-image/folkigpu.php
      - http://www.plyer.fr (author homepage)
*/

#include "LiGL2Dwidget.hpp"
namespace LIVA 
{

void 
LiGL2D::initializeGL()
{
	glewInit();
    if (! glewIsSupported(
        "GL_ARB_vertex_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
    }
	// default initialization
	qglClearColor((QColor::fromCmykF(0.39,0.39,0.0,0.0)).dark());
	glDisable(GL_DEPTH_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}


void
LiGL2D::draw()
{
	paintGL();
}

void
LiGL2D::save(QString url)
{
	char nom[256];
	sprintf(nom,"folkisave/%03d.jpg",_index);
	std::cout << "ecris :" << nom << " \n";

	_index++;
//	QPixmap pixmap = renderPixmap(this->size().width(),this->size().height());
//	pixmap.save(QString(nom));
	QImage tmp = grabFrameBuffer();
	tmp.save(QString(nom));
}

void
LiGL2D::setDrawIm(bool val)
{
	drawIm = val;
}

void
LiGL2D::paintEvent(QPaintEvent *)
{
	paintGL();
}

void
LiGL2D::paintGL()
{
	GLsizei w = (GLsizei)(this->size().width());
	GLsizei h = (GLsizei)(this->size().height());
	makeCurrent();
	if(initialise){
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
		if(drawIm){
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
			glBindTexture(GL_TEXTURE_2D, tex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
						iw, ih, 
						GL_BGRA, GL_UNSIGNED_BYTE, NULL);
			glDisable(GL_DEPTH_TEST);
			glDisable(GL_LIGHTING);
			glEnable(GL_TEXTURE_2D);
			glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
			glMatrixMode( GL_MODELVIEW);
			glLoadIdentity();
			glViewport(0, 0, w, h);
			glBegin(GL_QUADS);
				glTexCoord2f(0.0, 0.0);
				glVertex3f(-1.0, 1.0, 0.0);
				glTexCoord2f(1.0, 0.0);
				glVertex3f(1.0, 1.0, 0.0);
				glTexCoord2f(1.0, 1.0);
				glVertex3f(1.0, -1.0, 0.0);
				glTexCoord2f(0.0, 1.0);
				glVertex3f(-1.0, -1.0, 0.0);
			glEnd();
			glMatrixMode(GL_PROJECTION);
			glPopMatrix();
			glDisable(GL_TEXTURE_2D);
			glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			glDisable(GL_TEXTURE_2D);
		}
		if(vbo !=0 && drawVect ){
			glMatrixMode( GL_MODELVIEW);
			glLoadIdentity();
			glTranslatef(-1.0f,1.0f,0.0f);
			glScalef(2.0f/(float)imW,-2.0f/(float)imH,1.0f);
			glViewport(0, 0, w, h);
			// render from the vbo
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glVertexPointer(2, GL_FLOAT, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);
			glDisable(GL_DEPTH_TEST);
			glDisable(GL_LIGHTING);
			glColor3f(1.0, 0.0, 0.0);
			glDrawArrays(GL_LINES, 0, sVbo);
			glDisableClientState(GL_VERTEX_ARRAY);
			glBindBuffer( GL_ARRAY_BUFFER, 0);
		}
		if(partbo != 0 && drawPart){
			glMatrixMode( GL_MODELVIEW);
			glLoadIdentity();
			glTranslatef(-1.0f,1.0f,0.0f);
			glScalef(2.0f/(float)imW,-2.0f/(float)imH,1.0f);
			glViewport(0, 0, w, h);

		}
	}else{
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	}
	swapBuffers();
	glFlush();
}

void 
LiGL2D::setAffFlot(bool val)
{
	drawVect = val;
}



void
LiGL2D::setVbo(int spaceVect)
{
	GLuint oldVbo = 0;
	GLuint newVbo = 0;
	if(vbo != 0){
		oldVbo = vbo;
		vbo = 0;
	}
	if(iw != 0 && ih !=0){
		GLint bsize;
		// create buffer object
		unsigned int size = ((int)iw/(spaceVect+1))*((int)ih/(spaceVect+1)) * 6 *  sizeof(float2);
		glGenBuffers( 1, &newVbo);
		glBindBuffer( GL_ARRAY_BUFFER, newVbo);
		// initialize buffer object
		glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
		glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bsize); 
		glBindBuffer( GL_ARRAY_BUFFER, 0);
		// register buffer object with CUDA
		CUDA_SAFE_CALL(cudaGLRegisterBufferObject(newVbo));
		sVbo = ((int)iw/(spaceVect+1))*((int)ih/(spaceVect+1))*6;
		vbo = newVbo;
		emit sendVbo(vbo);
	}
	if(oldVbo != 0){
		CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(oldVbo));
		glDeleteBuffers(1, &oldVbo);
	}
}


void
LiGL2D::setPbo(int image_width, int image_height)
{
	makeCurrent();
	iw = image_width;
	ih = image_height;
	GLuint oldPbo = 0;
	GLuint newPbo = 0;
	GLuint oldTex = 0;

	if(pbo != 0){
		oldPbo = pbo;
		pbo = 0;
		oldTex = tex;
	}
	if(iw != 0 && ih !=0){
		glGenBuffers(1, &newPbo);
		glBindBuffer(GL_ARRAY_BUFFER, newPbo);
		glBufferData(GL_ARRAY_BUFFER, image_height*image_width* 4*sizeof(GLubyte),NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		CUDA_SAFE_CALL(cudaGLRegisterBufferObject(newPbo));
		createTexture(&tex, iw, ih);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);	
		pbo = newPbo;
		emit sendPbo(pbo);
	}
	if(oldPbo != 0){
		CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(oldPbo));
		glDeleteBuffers(1, &oldPbo);
	}
	if(oldTex != 0){
		glDeleteTextures(1, &oldTex);
	}
}

void
LiGL2D::init(int image_width, int image_height)
{
	GLint bsize;
	iw = image_width;
	ih = image_height;
	imW = iw;
	imH = ih;
	makeCurrent();
	// allocation du pbo
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_ARRAY_BUFFER, pbo);
	glBufferData(GL_ARRAY_BUFFER, image_height*image_width* 4*sizeof(GLubyte),NULL, GL_DYNAMIC_DRAW);
	glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bsize); 
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo));
	setVbo(10);
	// allocation de la texture d'affichage
	createTexture(&tex, image_width, image_height);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	initialise = true;
	std::cout << "buffer pixel num " << pbo << " taille : " << bsize << "\n";
	emit sendPbo(pbo);
}

void
LiGL2D::createTexture( GLuint* tex_name, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_name);
    glBindTexture(GL_TEXTURE_2D, *tex_name);
    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // buffer data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

} // namespace LIVA

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

#ifndef __LI_GL2D_WIDGET_HPP
#define __LI_GL2D_WIDGET_HPP
#include "GestionBuffer.hpp"
#include <QtOpenGL>
#include <iostream>
#include "LiCuda.hpp"
//#include <cutil_gl_error.h>

namespace LIVA
{
class LiGL2D :  public QGLWidget
{
	Q_OBJECT
private:
	GLuint  pbo;
	GLuint vbo;
	GLuint partbo;
	GLuint tex;
	GLsizei sVbo;
	bool drawIm;
	bool drawVect;
	bool drawPart;
	bool initialise;
	int iw,ih;
	int imW,imH;
	int _index;
	void createTexture( GLuint* tex_name, unsigned int size_x, unsigned int size_y);
public:
	LiGL2D(QWidget *parent = 0):QGLWidget(parent){
		_index = 0;
		pbo = 0;
		partbo = 0;
		vbo = 0;
		initialise = false;
		drawVect = false;
		drawIm = false;
		drawPart = false;
	}
	~LiGL2D(){
		if(initialise){
			glBindBuffer(GL_ARRAY_BUFFER, pbo);
			glDeleteBuffers(1, &pbo);
			glDeleteTextures(1, &tex);
			pbo = 0;
		}
	}

public slots:
	void draw();
	void init(int image_width, int image_height);
	void setVbo(int spaceVect);
	void setPbo(int image_width, int image_height);
	void paintEvent(QPaintEvent *);
	void setAffFlot(bool);	
	void setDrawIm(bool);
	void save(QString);

signals:
	void sendPbo(uint);
	void sendVbo(uint);

protected:
	void initializeGL();
	void paintGL();
	void resizeGL();
};
} // namespace LIVA

	
#endif

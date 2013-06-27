#include "analyser.h"
#include "sift.h"

#include <QDebug>
#include <QImage>
#include <qglview.h>

#include <GL/gl.h>

struct Analyser::pimpl
{
  pimpl ()
    : analyser (Sift::analyser ())
    , matcher (Sift::matcher ())
  {
  }

  void analyse (QImage const &img);

  SiftGPU &analyser;
  SiftMatchGPU &matcher;
};


void
Analyser::pimpl::analyse (QImage const &input)
{
  //QImage img = QGLWidget::convertToGLFormat (input);
  //analyser.RunSIFT (img.width (), img.height (), img.bits (),
                    //GL_RGBA, GL_UNSIGNED_BYTE);
#if 0
  for (int i = 0; i < 2; i++)
    {
      glBegin (GL_LINES);
      glVertex3f (0, 0, 5);
      glVertex3f ((rand () % 640) / 120. - 640 / 240.,
                  (rand () % 480) / 120. - 480 / 240.,
                  -0.1);
      glEnd ();
    }
#endif
}


Analyser::Analyser ()
  : self (new pimpl)
{
}

Analyser::~Analyser ()
{
}

void
Analyser::operator () (QImage const &img)
{
  self->analyse (img);
}

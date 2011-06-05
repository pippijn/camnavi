#include "siftview.h"
#include "register.h"

#include <qgltexture2d.h>
#include <qglcube.h>
#include <qglbuilder.h>
#include <qglpainter.h>

#include <opencv2/core/core.hpp>

#include "imagefeeder.h"
#include "glplane.h"
#include "analyserthread.h"

QML_REGISTER_TYPE (SIFTView);

class SIFTView::pimpl
  : public QObject
{
  Q_OBJECT

public:
  explicit pimpl (QObject *parent);
  ~pimpl ();

  void draw (QGLPainter *painter);

private: // functions
  QGLSceneNode *makeCube ();

private: // data
  AnalyserThread sift;
  ImageFeeder images;
  QGLTexture2D tex;
  std::auto_ptr<QGLSceneNode> const cube;

private slots:
  void analyseNext ();
};

SIFTView::pimpl::pimpl (QObject *parent)
  : QObject (parent)
  , cube (makeCube ())
{
  connect (&sift, SIGNAL (ready ()),
           this, SLOT (analyseNext ()));
  sift.start ();
}

SIFTView::pimpl::~pimpl ()
{
  sift.terminate ();
}

void
SIFTView::pimpl::draw (QGLPainter *painter)
{
  cube->draw (painter);
}


QGLSceneNode *
SIFTView::pimpl::makeCube ()
{
  QGLBuilder builder;
  builder << QGL::Faceted << GLPlane ();
  QGLSceneNode *cube = builder.finalizedSceneNode ();

  QGLMaterial *mat = new QGLMaterial (this);
  mat->setTexture (&tex);
  cube->setMaterial (mat);

  cube->setEffect (QGL::LitDecalTexture2D);

  return cube;
}


void
SIFTView::pimpl::analyseNext ()
{
  QImage const &img = images.next ();
  tex.setImage (img);
  sift.setImage (img);
  qobject_cast<SIFTView *> (parent ())->update ();
}


SIFTView::SIFTView (QObject *parent)
  : QDeclarativeItem3D (parent)
  , self (new pimpl (this))
{
}

SIFTView::~SIFTView ()
{
}

void
SIFTView::drawItem (QGLPainter *painter)
{
  self->draw (painter);
}

#include "src/bin/reconstruct/siftview.moc"

/* Copyright © 2010 Pippijn van Steenhoven
 * See COPYING.AGPL for licence information.
 */
#pragma once

#include <memory>

#include <qdeclarativeitem3d.h>

class SIFTView
  : public QDeclarativeItem3D
{
  Q_OBJECT

public:
  explicit SIFTView (QObject *parent = 0);
  ~SIFTView ();

protected:
  virtual void drawItem (QGLPainter *painter);

private:
  struct pimpl;
  std::auto_ptr<pimpl> const self;
};

#include "glplane.h"

#include <QVector2D>

#include "qt3dglobal.h"
#include "qvector2darray.h"
#include "qglbuilder.h"
#include "qvector3darray.h"

static int const vertexDataLen = 6 * 4 * 3;

static float const scale = 2;

static float const W = 640. / 480 * scale;
static float const H = 480. / 480 * scale;
static float const D =   1. / 480 * scale;

static float const vertexData[vertexDataLen] = {
   W, -H,  D,
   W,  H,  D,
  -W,  H,  D,
  -W, -H,  D,

  -W, -H, -D,
  -W, -H,  D,
  -W,  H,  D,
  -W,  H, -D,

  -W,  H, -D,
  -W,  H,  D,
   W,  H,  D,
   W,  H, -D,

   W,  H, -D,
   W,  H,  D,
   W, -H,  D,
   W, -H, -D,

   W, -H, -D,
   W, -H,  D,
  -W, -H,  D,
  -W, -H, -D,

   W,  H, -D,
   W, -H, -D,
  -W, -H, -D,
  -W,  H, -D,
};

static int const texCoordDataLen = 4 * 2;

static float const texCoordData[texCoordDataLen] = {
  1, 0,
  1, 1,
  0, 1,
  0, 0,
};

QGLBuilder &
operator << (QGLBuilder &builder, GLPlane plane)
{
  QGeometryData op;

  QVector3DArray vrts = QVector3DArray::fromRawData (
    reinterpret_cast<QVector3D const *> (vertexData), vertexDataLen / 3);

  op.appendVertexArray (vrts);

  QVector2DArray texx = QVector2DArray::fromRawData (
    reinterpret_cast<QVector2D const *> (texCoordData), texCoordDataLen / 2);

  //for (int i = 0; i < 6; ++i)
    op.appendTexCoordArray (texx);

  builder.addQuads (op);
  return builder;
}

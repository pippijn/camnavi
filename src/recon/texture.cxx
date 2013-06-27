#include "texture.h"

#include <opencv2/core/core.hpp>

#include <GL/glu.h>

using cv::Mat;

texture::texture ()
{
  // allocate a texture name
  glGenTextures (1, &tex);
  select ();
  // select modulate to mix texture with color for shading
  glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  // when texture area is small, bilinear filter the closest mipmap
  glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                   GL_LINEAR_MIPMAP_NEAREST);
  // when texture area is large, bilinear filter the original
  glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // the texture wraps over at the edges (repeat)
  glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

texture::texture (texture &&other)
{
}

texture::~texture ()
{
  if (tex != 0)
    glDeleteTextures (1, &tex);
}

texture &
texture::operator = (texture &&other)
{
  if (this != &other)
    {
      tex = other.tex;
      other.tex = 0;
    }
  return *this;
}


void
texture::select () const
{
  // select our current texture
  glBindTexture (GL_TEXTURE_2D, tex);
}

void
texture::load (Mat const &mat)
{
  select ();

  cv::Size const &size = mat.size ();
  // build our texture mipmaps
  gluBuild2DMipmaps (GL_TEXTURE_2D, 3, size.width, size.height,
                     GL_LUMINANCE, GL_UNSIGNED_BYTE, mat.ptr ());
}

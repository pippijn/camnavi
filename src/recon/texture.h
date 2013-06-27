#pragma once

#include "cvfwd.h"

#include <GL/gl.h>

struct texture
{
  texture ();
  texture (texture const &other) = delete;
  texture (texture &&other);
  ~texture ();

  texture &operator = (texture const &other) = delete;
  texture &operator = (texture &&other);

  void select () const;
  void load (cv::Mat const &mat);

  GLuint tex;
};

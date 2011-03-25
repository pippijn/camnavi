#pragma once

#include <opencv/cv.h>

__BEGIN_DECLS

void fft2 (IplImage const *src, IplImage *dst);
void fft2shift (IplImage const *src, IplImage *dst);

__END_DECLS

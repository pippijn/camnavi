/*
 * Miscellaneous utility functions.
 *
 * Copyright (C) 2006-2010  Rob Hess <hess@eecs.oregonstate.edu>
 *
 * @version 1.1.2-20100521
 */

#include "utils.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>


/*************************** Function Definitions ****************************/


/*
 * Prints an error message and aborts the program.  The error message is
 * of the form "Error: ...", where the ... is specified by the \a format
 * argument
 *
 * @param format an error message format string (as with \c printf(3)).
 */
void
fatal_error (char *format, ...)
{
  va_list ap;

  fprintf (stderr, "Error: ");

  va_start (ap, format);
  vfprintf (stderr, format, ap);
  va_end (ap);
  fprintf (stderr, "\n");
  abort ();
}

/*
 * Replaces a file's extension, which is assumed to be everything after the
 * last dot ('.') character.
 *
 * @param file the name of a file
 *
 * @param extn a new extension for \a file; should not include a dot (i.e.
 *  \c "jpg", not \c ".jpg") unless the new file extension should contain
 *  two dots.
 *
 * @return Returns a new string formed as described above.  If \a file does
 *  not have an extension, this function simply adds one.
 */
char *
replace_extension (const char *file, const char *extn)
{
  char *new_file, *lastdot;

  new_file = calloc (strlen (file) + strlen (extn) + 2, sizeof (char));
  strcpy (new_file, file);
  lastdot = strrchr (new_file, '.');
  if (lastdot)
    *(lastdot + 1) = '\0';
  else
    strcat (new_file, ".");
  strcat (new_file, extn);

  return new_file;
}

/*
 * Prepends a path to a filename.
 *
 * @param path a path
 * @param file a file name
 *
 * @return Returns a new string containing a full path name consisting of
 *  \a path prepended to \a file.
 */
char *
prepend_path (const char *path, const char *file)
{
  int n = strlen (path) + strlen (file) + 2;
  char *pathname = calloc (n, sizeof (char));

  snprintf (pathname, n, "%s/%s", path, file);

  return pathname;
}

/*
 * A function that removes the path from a filename.  Similar to the Unix
 * basename command.
 *
 * @param pathname a (full) path name
 *
 * @return Returns the basename of \a pathname.
 */
char *
basename (const char *pathname)
{
  char *base, *last_slash;

  last_slash = strrchr (pathname, '/');
  if (!last_slash)
    {
      base = calloc (strlen (pathname) + 1, sizeof (char));
      strcpy (base, pathname);
    }
  else
    {
      base = calloc (strlen (last_slash++), sizeof (char));
      strcpy (base, last_slash);
    }

  return base;
}

/*
 * Displays progress in the console with a spinning pinwheel.  Every time this
 * function is called, the state of the pinwheel is incremented.  The pinwheel
 * has four states that loop indefinitely: '|', '/', '-', '\'.
 *
 * @param done if 0, this function simply increments the state of the pinwheel;
 *  otherwise it prints "done"
 */
void
progress (int done)
{
  char state[4] = {
    '|', '/', '-', '\\'
  };
  static int cur = -1;

  if (cur == -1)
    fprintf (stderr, "  ");

  if (done)
    {
      fprintf (stderr, "\b\bdone\n");
      cur = -1;
    }
  else
    {
      cur = (cur + 1) % 4;
      fprintf (stdout, "\b\b%c ", state[cur]);
      fflush (stderr);
    }
}

/*
 * Erases a specified number of characters from a stream.
 *
 * @param stream the stream from which to erase characters
 * @param n the number of characters to erase
 */
void
erase_from_stream (FILE *stream, int n)
{
  int j;

  for (j = 0; j < n; j++)
    fprintf (stream, "\b");
  for (j = 0; j < n; j++)
    fprintf (stream, " ");
  for (j = 0; j < n; j++)
    fprintf (stream, "\b");
}

/*
 * Doubles the size of an array with error checking
 *
 * @param array pointer to an array whose size is to be doubled
 * @param n number of elements allocated for \a array
 * @param size size in bytes of elements in \a array
 *
 * @return Returns the new number of elements allocated for \a array.  If no
 *  memory is available, returns 0.
 */
int
array_double (void **array, int n, int size)
{
  void *tmp;

  tmp = realloc (*array, 2 * n * size);
  if (!tmp)
    {
      fprintf (stderr, "Warning: unable to allocate memory in array_double(),"
               " %s line %d\n", __FILE__, __LINE__);
      if (*array)
        free (*array);
      *array = NULL;
      return 0;
    }
  *array = tmp;
  return n * 2;
}

/*
 * Calculates the squared distance between two points.
 *
 * @param p1 a point
 * @param p2 another point
 */
double
dist_sq_2D (CvPoint2D64f p1, CvPoint2D64f p2)
{
  double x_diff = p1.x - p2.x;
  double y_diff = p1.y - p2.y;

  return x_diff * x_diff + y_diff * y_diff;
}

/*
 * Draws an x on an image.
 *
 * @param img an image
 * @param pt the center point of the x
 * @param r the x's radius
 * @param w the x's line weight
 * @param color the color of the x
 */
void
draw_x (IplImage *img, CvPoint pt, int r, int w, CvScalar color)
{
  cvLine (img, pt, cvPoint (pt.x + r, pt.y + r), color, w, 8, 0);
  cvLine (img, pt, cvPoint (pt.x - r, pt.y + r), color, w, 8, 0);
  cvLine (img, pt, cvPoint (pt.x + r, pt.y - r), color, w, 8, 0);
  cvLine (img, pt, cvPoint (pt.x - r, pt.y - r), color, w, 8, 0);
}

/*
 * Combines two images by scacking one on top of the other
 *
 * @param img1 top image
 * @param img2 bottom image
 *
 * @return Returns the image resulting from stacking \a img1 on top if \a img2
 */
extern IplImage *
stack_imgs (IplImage *img1, IplImage *img2)
{
  IplImage *stacked = cvCreateImage (cvSize (MAX (img1->width, img2->width),
                                             img1->height + img2->height),
                                     IPL_DEPTH_8U, 3);

  cvZero (stacked);
  cvSetImageROI (stacked, cvRect (0, 0, img1->width, img1->height));
  cvAdd (img1, stacked, stacked, NULL);
  cvSetImageROI (stacked, cvRect (0, img1->height, img2->width, img2->height));
  cvAdd (img2, stacked, stacked, NULL);
  cvResetImageROI (stacked);

  return stacked;
}

/*
 * Checks if a HighGUI window is still open or not
 *
 * @param name the name of the window we're checking
 *
 * @return Returns 1 if the window named \a name has been closed or 0 otherwise
 */
int
win_closed (char *win_name)
{
  if (!cvGetWindowHandle (win_name))
    return 1;
  return 0;
}

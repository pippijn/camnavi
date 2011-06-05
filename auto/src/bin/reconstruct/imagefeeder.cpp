#include "imagefeeder.h"

#include <cstdio>
#include <vector>

#include <QImage>

struct ImageFeeder::pimpl
{
  pimpl ();
  ~pimpl ();

  QImage const &next ();

  std::vector<QImage> imgs;
  std::vector<QImage>::iterator cur;
};

ImageFeeder::pimpl::pimpl ()
{
  for (int i = 0; i <= 395; i++)
    {
      char buf[PATH_MAX];
      snprintf (buf, sizeof buf, "../../images/%03d.jpg", i);
      imgs.push_back (QImage (buf));
    }
  cur = imgs.begin ();
}

ImageFeeder::pimpl::~pimpl ()
{
}

QImage const &
ImageFeeder::pimpl::next ()
{
  if (cur == imgs.end ())
    cur = imgs.begin ();
  return *cur++;
}


ImageFeeder::ImageFeeder ()
  : self (new pimpl)
{
}

ImageFeeder::~ImageFeeder ()
{
}

QImage const &
ImageFeeder::next ()
{
  return self->next ();
}

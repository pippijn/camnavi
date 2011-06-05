#include "gpusift.h"

#include <cstdio>
#include <map>

#include <boost/array.hpp>
#include <boost/circular_buffer.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <GL/gl.h>

#include "siftgpu/SiftGPU.h"
#include "timer.h"
#include "foreach.h"

using cv::Mat;

typedef SiftGPU::SiftKeypoint keypoint;

static SiftGPU *
init_sift ()
{
  SiftGPU *sift = new SiftGPU;
  char const *argv[] = {
    "-t", "0.01",
    "-v", "0",
    //"-cuda",
  };
  size_t argc = sizeof argv / sizeof *argv;
  sift->ParseParam (argc, const_cast<char **> (argv));
  if (sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
    {
      delete sift;
      throw 0;
    }
  return sift;
}

static std::auto_ptr<SiftGPU> sift;
static std::auto_ptr<SiftMatchGPU> matcher;


struct gpusift::pimpl
{
  static size_t const history_size = 6;

  struct history_item
  {
    static size_t last_id;

    static size_t next_id ()
    {
      return ++last_id;
    }

    size_t id;

    Mat mat;
    std::vector<keypoint> keys;
    std::vector<float> descriptors;

    bool empty () const
    {
      return mat.empty ();
    }

    history_item (Mat const &mat = Mat (), int count = 0)
      : id (next_id ())
      , mat (mat)
      , keys (count)
      , descriptors (128 * count)
    {
    }
  };
  
  struct history
  {
    size_t size () const { return data.size (); }
    history_item       &operator [] (size_t i)       { return data[i]; }
    history_item const &operator [] (size_t i) const { return data[i]; }
    history_item const &back () const { return data.back (); }
    history_item       &back ()       { return data.back (); }

    // returns true if the history is complete (all elements have been filled)
    bool push (history_item &newitem)
    {
      data.push_back (newitem);
      return data.size () == data.capacity ();
    }

    void print_ids () const
    {
      fputs ("history:", stdout);
      foreach (history_item const &item, data)
        printf (" %lu", item.id);
      fputc ('\n', stdout);
    }

    history ()
      : data (history_size)
    {
    }

    // TODO: replace with statically sized buffer
    boost::circular_buffer<history_item> data;
  } hist;

  void gpusift (Mat const &src, Mat &dst);
  static void draw (std::vector<keypoint> const &keys, Mat &dst);
  static void match (history_item const &item0, history_item const &item1, MatchBuffer &match_buffer);
  void match_all (std::map<int, MatchBuffer> &matches) const;
  void match (Mat &dst);
};

size_t gpusift::pimpl::history_item::last_id;

void
gpusift::pimpl::gpusift (Mat const &src, Mat &dst)
{
  timer const T (__func__);
  if (!sift.get ())
    {
      sift.reset (init_sift ());
      matcher.reset (new SiftMatchGPU (4096));
      matcher->VerifyContextGL ();
    }

  cv::Size const &size = src.size ();
  if (!sift->RunSIFT (size.width, size.height, src.ptr (), GL_LUMINANCE, GL_UNSIGNED_BYTE))
    throw 0;

  int count = sift->GetFeatureNum ();
  // TODO: don't create new ones all the time
  history_item cur (src, count);
  sift->GetFeatureVector (&cur.keys[0], &cur.descriptors[0]);

  if (hist.push (cur))
    match (dst);
  else
    draw (cur.keys, dst);
}

void
gpusift::pimpl::draw (std::vector<keypoint> const &keys, Mat &dst)
{
  foreach (keypoint const &p, keys)
    dst.at<uchar> (p.y, p.x) = 255;
}

void
gpusift::pimpl::match (history_item const &item0, history_item const &item1, MatchBuffer &match_buffer)
{
  //printf ("matching %lu and %lu: ", item0.id, item1.id);
  matcher->SetDescriptors (0, item0.keys.size (), &item0.descriptors[0]);
  matcher->SetDescriptors (1, item1.keys.size (), &item1.descriptors[0]);

  matcher->GetSiftMatch (match_buffer);
  //printf ("%lu matches\n", match_buffer.size ());
}

void
gpusift::pimpl::match_all (std::map<int, MatchBuffer> &matches) const
{
  history_item const &last = hist.back ();

  // match all items with all other items
  foreach (history_item const &ref, hist.data)
    {
      if (&ref != &last)
        {
          MatchBuffer &match_buffer = matches[ref.id];
          match (ref, last, match_buffer);
        }
    }
}

template<int W, int H>
struct compare_xy
{
  bool operator () (keypoint const &a, keypoint const &b)
  {
    return (int)a.y * W + (int)a.x
         < (int)b.y * W + (int)b.x
         ;
  }
};

void
gpusift::pimpl::match (Mat &dst)
{
  //hist.print_ids ();
  //return;

  std::map<int, MatchBuffer> matches;

  match_all (matches);

  history_item const &last = hist.back ();
  std::map<keypoint, size_t, compare_xy<640, 480> > good;

  // go through each item's match set and remove the ones that are not in all
  // match sets for the target
  // TODO
  foreach (history_item &ref, hist.data)
    {
      if (ref.id == last.id - 1)
        break;

      MatchBuffer &match_buffer = matches[ref.id];
      for (size_t m = 0; m < match_buffer.size (); ++m)
        {
          keypoint const &key0 = ref.keys[match_buffer[m][0]];
          keypoint const &key1 = last.keys[match_buffer[m][1]];
          if (good[key1]++)
            printf ("%lu -> [%g, %g] %g, %g\n", ref.id, key1.x, key1.y, key1.s, key1.o);
        }
    }

  dst = Mat::zeros (dst.size (), CV_8UC3);
  MatchBuffer &match_buffer = matches[last.id - 1];
  for (size_t m = 0; m < match_buffer.size (); ++m)
    {
      keypoint const &key0 = hist[history_size - 2].keys[match_buffer[m][0]];
      keypoint const &key1 = hist[history_size - 1].keys[match_buffer[m][1]];
      bool const is_good = good[key1] > history_size / 3;
      line (dst,
            cv::Point (key0.x, key0.y),
            cv::Point (key1.x, key1.y),
            cv::Scalar (255 * is_good, 255 * is_good, 255));
    }
}


gpusift::gpusift ()
  : self (new pimpl)
{
}

gpusift::~gpusift ()
{
}

void
gpusift::operator () (Mat const &src, Mat &dst)
{
  self->gpusift (src, dst);
}

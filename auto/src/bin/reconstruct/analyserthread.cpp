#include "analyserthread.h"

#include <cassert>

#include <QDebug>

AnalyserThread::AnalyserThread (QObject *parent)
  : QThread (parent)
  , image (0)
{
  mutex.lock ();
}

AnalyserThread::~AnalyserThread ()
{
}

void
AnalyserThread::run ()
{
  forever
    {
      emit ready ();
      mutex.lock ();
      assert (image != NULL);
      analyser (*image);
      image = NULL;
    }
}

void
AnalyserThread::setImage (QImage const &img)
{
  assert (image == NULL);
  image = &img;
  mutex.unlock ();
}

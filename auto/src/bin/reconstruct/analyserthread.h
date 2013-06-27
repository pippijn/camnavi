#pragma once

#include <memory>

#include <QMutex>
#include <QObject>
#include <QThread>
#include <QImage>

#include "analyser.h"

class QImage;

class AnalyserThread
  : public QThread
{
  Q_OBJECT

public:
  AnalyserThread (QObject *parent = 0);
  ~AnalyserThread ();

  // starts an infinite loop of analysing images
  void run ();

  // adds an image to the queue
  void setImage (QImage const &img);

signals:
  void ready ();

private:
  QImage const *volatile image;
  QMutex mutex;
  Analyser analyser;
};

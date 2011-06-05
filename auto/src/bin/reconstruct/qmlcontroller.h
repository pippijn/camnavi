/* Copyright © 2010 Pippijn van Steenhoven
 * See COPYING.AGPL for licence information.
 */
#pragma once

#include <QObject>

class QDeclarativeView;

class QmlController
  : public QObject
{
  Q_OBJECT

public:
  explicit QmlController (QDeclarativeView &view, QObject *parent = 0);
  ~QmlController ();

private:
  QDeclarativeView &m_view;
};

/* Copyright © 2010 Pippijn van Steenhoven
 * See COPYING.AGPL for licence information.
 */
#include "qmlcontroller.h"

#include <QDeclarativeContext>
#include <QDeclarativeEngine>
#include <QDeclarativeView>

QmlController::QmlController (QDeclarativeView &view, QObject *parent)
  : QObject (parent)
  , m_view (view)
{
}

QmlController::~QmlController ()
{
}

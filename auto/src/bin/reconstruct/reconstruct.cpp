/* Copyright © 2010 Pippijn van Steenhoven
 * See COPYING.AGPL for licence information.
 */
#include <QApplication>
#include <QDeclarativeContext>
#include <QDeclarativeEngine>
#include <QDeclarativeView>

#include "qmlcontroller.h"

int
main (int argc, char **argv)
{
  QApplication qca (argc, argv);

  qca.setApplicationName ("Groovy");

  QStringList args = qca.arguments ();

  // View
  QDeclarativeView view;
  view.setResizeMode (QDeclarativeView::SizeRootObjectToView);

  // Controller
  QmlController controller (view);

  // Linking model and view
  QDeclarativeContext *context = view.rootContext ();
  context->setContextProperty ("controller", &controller);

  // Load UI description into view
  view.setSource (QUrl ("qrc:/views/mainwindow.qml"));

  // Allow QML application to quit the entire application
  QObject::connect (view.engine (), SIGNAL (quit ()), &qca, SLOT (quit ()));

  // Run the application
  view.show ();
  qca.exec ();
}

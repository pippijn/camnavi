#pragma once

#include <QtDeclarative>

template<typename T>
struct QMLTypeRegisterer
{
  QMLTypeRegisterer (char const *name)
  {
    qmlRegisterType<T> ("Reconstructor", 1, 0, name);
  }
};

#define QML_REGISTER_TYPE(type) \
  QML_DECLARE_TYPE (type)       \
  static QMLTypeRegisterer<type> type##_REGISTER (#type)

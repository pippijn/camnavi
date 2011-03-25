/****************************************************************************
** Meta object code from reading C++ file 'Manager.h'
**
** Created: Fri Mar 25 11:22:32 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "Manager.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Manager.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_rec__robotino__imagesender__Manager[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      37,   36,   36,   36, 0x05,

 // slots: signature, parameters, type, tag, flags
     111,   61,   36,   36, 0x0a,
     169,  164,   36,   36, 0x0a,
     197,   61,   36,   36, 0x0a,
     255,  164,   36,   36, 0x0a,
     301,  288,   36,   36, 0x0a,
     330,  288,   36,   36, 0x0a,
     362,   36,   36,   36, 0x0a,
     370,   36,   36,   36, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_rec__robotino__imagesender__Manager[] = {
    "rec::robotino::imagesender::Manager\0"
    "\0imageSendingCompleted()\0"
    "data,width,height,numChannels,bitsPerChannel,step\0"
    "setRawImageData(QByteArray,uint,uint,uint,uint,uint)\0"
    "data\0setJpgImageData(QByteArray)\0"
    "setLocalRawImageData(QByteArray,uint,uint,uint,uint,uint)\0"
    "setLocalJpgImageData(QByteArray)\0"
    "address,port\0addReceiver(quint32,quint16)\0"
    "removeReceiver(quint32,quint16)\0reset()\0"
    "on_sender_imageSendingCompleted()\0"
};

const QMetaObject rec::robotino::imagesender::Manager::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_rec__robotino__imagesender__Manager,
      qt_meta_data_rec__robotino__imagesender__Manager, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &rec::robotino::imagesender::Manager::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *rec::robotino::imagesender::Manager::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *rec::robotino::imagesender::Manager::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_rec__robotino__imagesender__Manager))
        return static_cast<void*>(const_cast< Manager*>(this));
    return QObject::qt_metacast(_clname);
}

int rec::robotino::imagesender::Manager::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: imageSendingCompleted(); break;
        case 1: setRawImageData((*reinterpret_cast< const QByteArray(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< uint(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4])),(*reinterpret_cast< uint(*)>(_a[5])),(*reinterpret_cast< uint(*)>(_a[6]))); break;
        case 2: setJpgImageData((*reinterpret_cast< const QByteArray(*)>(_a[1]))); break;
        case 3: setLocalRawImageData((*reinterpret_cast< const QByteArray(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< uint(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4])),(*reinterpret_cast< uint(*)>(_a[5])),(*reinterpret_cast< uint(*)>(_a[6]))); break;
        case 4: setLocalJpgImageData((*reinterpret_cast< const QByteArray(*)>(_a[1]))); break;
        case 5: addReceiver((*reinterpret_cast< quint32(*)>(_a[1])),(*reinterpret_cast< quint16(*)>(_a[2]))); break;
        case 6: removeReceiver((*reinterpret_cast< quint32(*)>(_a[1])),(*reinterpret_cast< quint16(*)>(_a[2]))); break;
        case 7: reset(); break;
        case 8: on_sender_imageSendingCompleted(); break;
        default: ;
        }
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void rec::robotino::imagesender::Manager::imageSendingCompleted()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE

/****************************************************************************
** Meta object code from reading C++ file 'Sender.h'
**
** Created: Fri Mar 25 11:22:07 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "Sender.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Sender.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_rec__robotino__imagesender__Sender[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      36,   35,   35,   35, 0x05,

 // slots: signature, parameters, type, tag, flags
     110,   60,   35,   35, 0x0a,
     168,  163,   35,   35, 0x0a,
     209,  196,   35,   35, 0x0a,
     238,   35,   35,   35, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_rec__robotino__imagesender__Sender[] = {
    "rec::robotino::imagesender::Sender\0\0"
    "imageSendingCompleted()\0"
    "data,width,height,numChannels,bitsPerChannel,step\0"
    "setRawImageData(QByteArray,uint,uint,uint,uint,uint)\0"
    "data\0setJpgImageData(QByteArray)\0"
    "address,port\0setReceiver(quint32,quint16)\0"
    "on_timer_timeout()\0"
};

const QMetaObject rec::robotino::imagesender::Sender::staticMetaObject = {
    { &QUdpSocket::staticMetaObject, qt_meta_stringdata_rec__robotino__imagesender__Sender,
      qt_meta_data_rec__robotino__imagesender__Sender, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &rec::robotino::imagesender::Sender::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *rec::robotino::imagesender::Sender::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *rec::robotino::imagesender::Sender::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_rec__robotino__imagesender__Sender))
        return static_cast<void*>(const_cast< Sender*>(this));
    return QUdpSocket::qt_metacast(_clname);
}

int rec::robotino::imagesender::Sender::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QUdpSocket::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: imageSendingCompleted(); break;
        case 1: setRawImageData((*reinterpret_cast< const QByteArray(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< uint(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4])),(*reinterpret_cast< uint(*)>(_a[5])),(*reinterpret_cast< uint(*)>(_a[6]))); break;
        case 2: setJpgImageData((*reinterpret_cast< const QByteArray(*)>(_a[1]))); break;
        case 3: setReceiver((*reinterpret_cast< quint32(*)>(_a[1])),(*reinterpret_cast< quint16(*)>(_a[2]))); break;
        case 4: on_timer_timeout(); break;
        default: ;
        }
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void rec::robotino::imagesender::Sender::imageSendingCompleted()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE

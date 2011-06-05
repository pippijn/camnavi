#ifndef _REC_ROBOTINO_COM_V1_MESSAGES_INFO_H_
#define _REC_ROBOTINO_COM_V1_MESSAGES_INFO_H_

#include <QByteArray>
#include <QString>

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			namespace v1
			{
				namespace messages
				{
					class Info
					{
					public:
						Info( const QByteArray& data );

						QString text() const { return _text; }

						bool isPassiveMode() const { return _isPassiveMode; }

					private:
						QString _text;
						bool _isPassiveMode;
					};
				}
			}
		}
	}
}

#endif //_REC_ROBOTINO_COM_V1_MESSAGES_INFO_H_

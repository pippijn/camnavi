#ifndef _REC_CORE_LT_VERSION_H_
#define _REC_CORE_LT_VERSION_H_

#include <string>
#include <sstream>

#ifdef HAVE_QT
#include <QString>
#endif //HAVE_QT

namespace rec
{
	namespace core_lt
	{
		class Version
		{
		public:
			Version()
				: _major( 0 )
				, _minor( 0 )
				, _patch( 0 )
			{
			}

			Version( const char* version )
				: _major( 0 )
				, _minor( 0 )
				, _patch( 0 )
			{
				fromString( version );
			}

			Version( const std::string& version )
				: _major( 0 )
				, _minor( 0 )
				, _patch( 0 )
			{
				fromString( version.c_str() );
			}

#ifdef HAVE_QT
			Version( const QString& version )
				: _major( 0 )
				, _minor( 0 )
				, _patch( 0 )
			{
				fromString( version.toLatin1().constData() );
			}
#endif //HAVE_QT

			Version( int major, int minor, int patch )
				: _major( major )
				, _minor( minor )
				, _patch( patch )
			{
			}

			bool isEmpty() const
			{
				return ( (0 == _major) && (0 == _minor) && (0 == _patch) );
			}

			std::string toString() const
			{
				if( isEmpty() )
				{
					return "";
				}
				else
				{
					std::ostringstream os;
					os << _major << "." << _minor << "." << _patch;
					return os.str();
				}
			}

#ifdef HAVE_QT
			QString toQString() const
			{
				std::string str = toString();
				return QString::fromStdString( str );
			}
#endif //HAVE_QT

			void fromString( const char* version )
			{
				_major = -1;

				char ch;
				std::istringstream is( version );
				is >> _major;
				is.clear();

				while( false == is.eof() && -1 == _major )
				{
					is >> ch;
					if( is.eof() )
					{
						break;
					}

					is >> _major;
					is.clear();
				}

				if( -1 == _major )
				{
					reset();
					return;
				}

				is >> ch;
				if( is.good() && ( '.' == ch || '-' == ch ) )
				{
					is >> _minor;
					is >> ch;
					if( is.good() && ( '.' == ch || '-' == ch ) )
					{
						is >> _patch;
						if( is.fail() )
						{
							reset();
						}
					}
					else
					{
						reset();
					}
				}
				else
				{
					reset();
				}
			}

			void reset()
			{
				_major = 0;
				_minor = 0;
				_patch = 0;
			}

			bool operator > ( const Version& other ) const
			{
				if( _major > other._major )
				{
					return true;
				}
				else if( _major == other._major && _minor > other._minor )
				{
					return true;
				}
				else if( _major == other._major && _minor == other._minor && _patch > other._patch )
				{
					return true;
				}

				return false;
			}

			bool operator < ( const Version& other ) const
			{
				if( _major < other._major )
				{
					return true;
				}
				else if( _major == other._major && _minor < other._minor )
				{
					return true;
				}
				else if( _major == other._major && _minor == other._minor && _patch < other._patch )
				{
					return true;
				}

				return false;
			}

			bool operator == ( const Version& other ) const
			{
				if( _major == other._major && _minor == other._minor && _patch == other._patch )
				{
					return true;
				}

				return false;
			}

			int major() const { return _major; }
			
			int minor() const { return _minor; }
			
			int patch() const { return _patch; }

		private:
			int _major;
			int _minor;
			int _patch;
		};
	}
}

#ifdef HAVE_QT
#include <QMetaType>
Q_DECLARE_METATYPE( rec::core_lt::Version )
#endif

#endif //_REC_CORE_LT_VERSION_H_

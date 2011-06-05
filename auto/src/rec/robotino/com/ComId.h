//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_COMID_H_
#define _REC_ROBOTINO_COM_COMID_H_

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class ComImpl;

			/**
			\brief The id of a Com object.

			The ComId is passed to Actor object to establish a relation ship to the Com object.
			*/
			class
#ifdef WIN32
#  ifdef rec_robotino_com_EXPORTS
				__declspec(dllexport)
#  else
#    ifdef rec_robotino_com2_EXPORTS
				__declspec(dllexport)
#    else
#      ifdef rec_robotino_com3_EXPORTS
				__declspec(dllexport)
#      else
#        ifndef rec_robotino_com_static
				__declspec(dllimport)
#        endif
#      endif
#    endif
#  endif
#endif
			ComId
			{
				friend class ComImpl;
			public:
				ComId()
					: _id( 1 )
				{
				}

				bool operator==( const ComId& other ) const
				{
					return other._id == _id;
				}

				bool operator!=( const ComId& other ) const
				{
					return other._id != _id;
				}

				bool operator<( const ComId& other ) const
				{
					return other._id < _id;
				}

				bool isNull() const;

				operator bool() const { return !isNull(); }

				static const ComId null;

			private:
				ComId( unsigned int id )
					: _id( id )
				{
				}

				static unsigned int g_id;

				unsigned int _id;
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_COMID_H_

//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_SHAREDBASE_H_
#define _REC_CORE_LT_SHAREDBASE_H_

#include <boost/shared_ptr.hpp>

namespace rec
{
	namespace core_lt
	{
		template < class SharedTypeImpl >
		class SharedBase
		{
		public:

		protected:
			SharedBase( SharedTypeImpl* impl )
				: _impl( impl )
			{
			}

			void detach()
			{
				if( ! _impl.unique() )
				{
					boost::shared_ptr< SharedTypeImpl > ni( new SharedTypeImpl( *_impl ) );
					_impl = ni;
				}
			}

			boost::shared_ptr< SharedTypeImpl > _impl;
		};
	}
}

#endif //_REC_CORE_LT_SHAREDBASE_H_

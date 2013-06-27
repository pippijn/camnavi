//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_MEMORY_ALLOCATOR_H_
#define _REC_CORE_LT_MEMORY_ALLOCATOR_H_

namespace rec
{
	namespace core_lt
	{
		namespace memory
		{
			class Allocator
			{
			public:
				static unsigned char* malloc( unsigned int size );
				static unsigned char* realloc( unsigned char* buffer, unsigned int newSize );
				static void free( unsigned char* p );
				static void copy( unsigned char* dst, const unsigned char* src, unsigned int size );
				static bool isAligned32( const unsigned char* const p );
			private:
				Allocator() {}
				Allocator( const Allocator& ) {}
				void operator=( const Allocator& ) {}
			};
		}
	}
}

#endif //_REC_CORE_LT_MEMORY_ALLOCATOR_H_

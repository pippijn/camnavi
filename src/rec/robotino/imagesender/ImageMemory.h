#ifndef _REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_H_
#define _REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_H_

#include <string>

#define REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_RAWIMAGEDATASIZE 2359296 //1024*768*3
#define REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_JPGDATASIZE 500000
#define REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_KEY( serverPort ) QString("REC_ROBOTINO_IMAGESENDER_%1").arg( serverPort )

namespace rec
{
	namespace robotino
	{
		namespace imagesender
		{
			class ImageMemory
			{
			public:
				ImageMemory()
				{
				}

				ImageMemory( const ImageMemory& other )
				{
					copy( other );
				}

				ImageMemory& operator=( const ImageMemory& other )
				{
					copy( other );
					return *this;
				}

				void copy( const ImageMemory& other )
				{
					width = other.width;
					height = other.height;
					numChannels = other.numChannels;
					bitsPerChannel = other.bitsPerChannel;
					step = other.step;
					sequenceNumber = other.sequenceNumber;
					jpgSequenceNumber = other.jpgSequenceNumber;
					
					dataSize = other.dataSize;
					memcpy( data, other.data, dataSize );

					jpgDataSize = other.jpgDataSize;
					memcpy( jpgData, other.jpgData, jpgDataSize );

					numJpgRequests = other.numJpgRequests;
					numRawRequests = other.numRawRequests;
				}

				void reset()
				{
					width = 0;
					height = 0;
					numChannels = 0;
					bitsPerChannel = 0;
					step = 0;
					sequenceNumber = 0;
					jpgSequenceNumber = 0;
					
					dataSize = 0;
					memset( data, 0, REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_RAWIMAGEDATASIZE );

					jpgDataSize = 0;
					memset( jpgData, 0, REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_JPGDATASIZE );

					numJpgRequests = 0;
					numRawRequests = 0;
				}

				void registerJpgRequest( int num = 1 )
				{
					numJpgRequests += num;
				}

				void unregisterJpgRequest( int num = 1 )
				{
					if( num > numJpgRequests )
					{
						numJpgRequests = 0;
					}
					else
					{
						numJpgRequests -= num;
					}
				}

				void registerRawRequest( int num = 1 )
				{
					numRawRequests += num;
				}

				void unregisterRawRequest( int num = 1 )
				{
					if( num > numRawRequests )
					{
						numRawRequests = 0;
					}
					else
					{
						numRawRequests -= num;
					}
				}

				unsigned int width;
				unsigned int height;
				unsigned int numChannels;
				unsigned int bitsPerChannel;
				unsigned int step;
				unsigned int sequenceNumber;
				unsigned int jpgSequenceNumber;

				unsigned int dataSize;
				unsigned char data[REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_RAWIMAGEDATASIZE];

				unsigned int jpgDataSize;
				unsigned char jpgData[REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_JPGDATASIZE];

				int numJpgRequests;
				int numRawRequests;
			};
		}
	}
}

#endif //_REC_ROBOTINO_IMAGESENDER_IMAGEMEMORY_H_
